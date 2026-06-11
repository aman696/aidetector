"""
screenshot_classifier.py — Dedicated SVM Classifier for Screenshot Detection

A focused, lightweight SVM trained ONLY on screenshot data (AI screenshots vs
real screenshots). Uses 15 features specifically tuned for screen-rendered
content rather than camera-captured content.

Research basis:
  - Ng et al. (Imperial College) — multi-scale GLCM/LBP for recaptured images
  - Thongkamwitoon et al. — multi-scale wavelet pyramid for screen forensics
  - "Any-Resolution AI Detection by Spectral Learning" (arXiv Nov 2024)
    → FFT radial power slope survives VAE high-freq attenuation
  - SCI (Screen Content Image) vs natural image literature → histogram bimodality

WHY A SEPARATE CLASSIFIER:
  The main 79-feature SVM is trained on camera photos and downloaded files.
  For screenshots, ELA, DCT, PRNU noise, and metadata are all zero or noise
  (no JPEG grid from camera, no EXIF, no sensor noise). Mixing screenshot data
  into the main SVM pollutes it. A dedicated 15-feature / 35-sample model with
  a smaller feature-to-sample ratio generalises better.

Feature vector (15 features, indices and expected direction):
  Multi-scale GLCM (6):
    0. glcm_homogeneity_d1   — real UI: lower (structured text)
    1. glcm_homogeneity_d3   — same
    2. glcm_homogeneity_d7   — same
    3. glcm_contrast_d1      — real UI: higher (sharp text edges)
    4. glcm_contrast_d3      — same
    5. glcm_contrast_d7      — same

  Multi-level Haar wavelet (3):
    6. wavelet_hh_l1         — real UI: higher (fine diagonal edges)
    7. wavelet_hh_l2         — real UI: higher (mid-scale structure)
    8. wavelet_hh_l3         — real UI: higher (coarse icon structure)

  Multi-radius LBP entropy (2):
    9. lbp_entropy_r1        — AI: higher (large uniform smooth regions)
   10. lbp_entropy_r2        — same at coarser scale

  FFT radial power slope (1):
   11. fft_radial_slope      — AI: flatter falloff (VAE attenuation of HF)

  Histogram shape (2):
   12. hist_bimodality       — real UI: higher (dark bg + bright text)
   13. hist_peak_valley      — real UI: higher

  Chroma (1):
   14. chroma_std_ratio      — real UI: slightly higher
"""

import os
import numpy as np
import cv2
import joblib
from typing import Dict, List, Tuple, Optional, Any
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


# ──────────────────────────────────────────────────────────────────────────────
# Feature extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _glcm_stats(gray: np.ndarray, distance: int) -> Tuple[float, float]:
    """
    Compute GLCM homogeneity and contrast at a single distance across 4 angles.
    Returns (homogeneity, contrast), both averaged over angles.
    """
    q      = (gray // 8).astype(np.int32)   # quantise to 32 levels
    levels = 32

    def _one_angle(dx: int, dy: int):
        h, w = q.shape
        r0, r1 = max(0, -dy), min(h, h - dy)
        c0, c1 = max(0, -dx), min(w, w - dx)
        p1 = q[r0:r1, c0:c1]
        p2 = q[r0 + dy: r1 + dy, c0 + dx: c1 + dx]
        if p1.size == 0:
            return 0.0, 0.0

        glcm = np.zeros((levels, levels), dtype=np.float64)
        np.add.at(glcm, (p1.ravel(), p2.ravel()), 1)
        glcm = glcm + glcm.T
        total = glcm.sum()
        if total < 1:
            return 0.0, 0.0
        glcm /= total

        ii, jj = np.meshgrid(np.arange(levels), np.arange(levels), indexing='ij')
        homo  = float((glcm / (1 + np.abs(ii - jj))).sum())
        contr = float((glcm * (ii - jj)**2).sum())
        return homo, contr

    offsets = [(distance, 0), (0, distance),
               (distance,  distance), (distance, -distance)]
    homos, contrs = [], []
    for dx, dy in offsets:
        h, c = _one_angle(dx, dy)
        homos.append(h)
        contrs.append(c)

    return float(np.mean(homos)), float(np.mean(contrs))


def _lbp_entropy(gray: np.ndarray, radius: int) -> float:
    """
    Shannon entropy of LBP histogram at given radius.
    Returns value in [0, 1] (normalised by 8 bits).
    """
    h, w = gray.shape
    if min(h, w) < 2 * radius + 4:
        return 0.0

    center    = gray[radius: h - radius, radius: w - radius].astype(np.int16)
    lbp       = np.zeros_like(center, dtype=np.uint8)

    angles    = 8
    step      = 2 * np.pi / angles
    for bit in range(angles):
        angle = bit * step
        dy    = int(round(radius * np.sin(angle)))
        dx    = int(round(radius * np.cos(angle)))
        r0, r1 = radius + dy, h - radius + dy
        c0, c1 = radius + dx, w - radius + dx
        neighbour = gray[r0:r1, c0:c1].astype(np.int16)
        lbp |= ((neighbour >= center).astype(np.uint8) << bit)

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist     = hist.astype(np.float64) / (hist.sum() + 1e-10)
    entropy  = -np.sum(hist * np.log2(hist + 1e-10))
    return float(np.clip(entropy / 8.0, 0, 1))


def _haar_hh_energy(gray: np.ndarray, levels: int = 3) -> List[float]:
    """
    Multi-level Haar DWT. Returns log(HH energy) for each level.
    Level 1 = finest detail, level N = coarsest.
    """
    img    = gray.astype(np.float32)
    result = []
    for _ in range(levels):
        h, w = img.shape
        img  = img[:h - (h % 2), :w - (w % 2)]
        if min(img.shape) < 4:
            result.append(0.0)
            continue

        h, w        = img.shape
        re          = img[0::2, :]
        ro          = img[1::2, :]
        low_row     = (re + ro) / 2.0
        high_row    = (re - ro) / 2.0
        HH          = (high_row[:, 0::2] - high_row[:, 1::2]) / 2.0
        LL          = (low_row[:, 0::2]  + low_row[:, 1::2])  / 2.0

        result.append(float(np.log1p(np.mean(HH**2))))
        img         = LL         # descend to next level

    return result


def _fft_radial_slope(gray: np.ndarray) -> float:
    """
    Compute the slope of the log-log radial power spectrum.

    AI images (VAE-based) have a flatter slope because the VAE bottleneck
    attenuates high-frequency energy, causing energy to fall off more slowly
    than in real images where natural 1/f^2 statistics are preserved.

    Returns the slope (negative — steeper = more real, flatter = more AI).
    Normalised to range [-1, 0]: -1 = very steep (real), 0 = flat (AI).
    """
    side = min(gray.shape[0], gray.shape[1], 256)
    ch, cw = gray.shape[0] // 2, gray.shape[1] // 2
    img  = gray[ch - side//2: ch + side//2, cw - side//2: cw + side//2]

    fft   = np.fft.fft2(img.astype(np.float32))
    fft_s = np.fft.fftshift(fft)
    mag   = np.abs(fft_s) + 1.0

    cy2, cx2 = side // 2, side // 2
    ys, xs   = np.ogrid[-cy2: side - cy2, -cx2: side - cx2]
    r        = np.sqrt(xs**2 + ys**2).astype(np.float32)

    # Bin into 20 radial rings, compute mean log power per ring
    max_r  = float(cy2)
    bins   = np.linspace(1, max_r, 21)
    log_r, log_p = [], []
    for i in range(len(bins) - 1):
        mask = (r >= bins[i]) & (r < bins[i+1])
        if mask.sum() < 5:
            continue
        log_r.append(float(np.log(bins[i])))
        log_p.append(float(np.log(mag[mask].mean())))

    if len(log_r) < 4:
        return 0.0

    # Fit slope via least squares
    log_r_arr = np.array(log_r)
    log_p_arr = np.array(log_p)
    A = np.stack([log_r_arr, np.ones_like(log_r_arr)], axis=1)
    slope, _ = np.linalg.lstsq(A, log_p_arr, rcond=None)[0]

    # Typical range: real images -2.5 to -1.0, AI images -1.5 to -0.5
    # Normalise so that -2.5 → -1.0, -0.5 → 0.0
    return float(np.clip((slope + 0.5) / (-2.0), -1.0, 0.0))


def _histogram_shape(gray: np.ndarray) -> Tuple[float, float]:
    """
    Compute bimodality coefficient and peak/valley ratio of the intensity
    histogram.

    Real UI screenshots have bimodal histograms: dark background + bright
    text/icons. AI images have smoother bell-curve distributions.

    Bimodality coefficient (Sarle 1948):
        BC = (skewness^2 + 1) / kurtosis
        BC > 5/9 ~ 0.555 suggests bimodal distribution.

    Returns (bimodality_coeff, peak_valley_ratio) both in [0, 1].
    """
    h, _ = np.histogram(gray.ravel(), bins=64, range=(0, 256))
    h     = h.astype(np.float64)

    # Bimodality via skewness and kurtosis
    n    = float(gray.size)
    if n < 4:
        return 0.0, 0.0

    mean  = gray.mean()
    std   = gray.std() + 1e-8
    skew  = float(np.mean(((gray.astype(np.float64) - mean) / std)**3))
    kurt  = float(np.mean(((gray.astype(np.float64) - mean) / std)**4))
    if kurt < 1e-8:
        bc = 0.0
    else:
        bc = float((skew**2 + 1.0) / kurt)
    # Normalise: BC in [0, ~2], typical bimodal: 0.6–1.2, unimodal: 0.2–0.5
    bc_norm = float(np.clip(bc / 1.5, 0, 1))

    # Peak/valley ratio: ratio of max to min in histogram (ignoring tails)
    inner   = h[4:60]                       # skip extreme tails
    h_norm  = inner / (inner.max() + 1e-10)
    peaks   = (np.diff(np.sign(np.diff(h_norm))) < 0).sum()
    valleys = (np.diff(np.sign(np.diff(h_norm))) > 0).sum()
    pv      = float(np.clip((peaks + valleys) / 20.0, 0, 1))

    return bc_norm, pv


def _chroma_std_ratio(img_bgr: np.ndarray) -> float:
    """Imbalance between Cb and Cr std dev in YCbCr space."""
    ycrcb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    std_cb = float(np.std(ycrcb[:, :, 2]))
    std_cr = float(np.std(ycrcb[:, :, 1]))
    return float(np.clip(abs(std_cb / (std_cr + 1e-8) - 1.0), 0, 5))


# ──────────────────────────────────────────────────────────────────────────────
# Public extraction API
# ──────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # Multi-scale GLCM
    'glcm_homogeneity_d1', 'glcm_homogeneity_d3', 'glcm_homogeneity_d7',
    'glcm_contrast_d1',    'glcm_contrast_d3',    'glcm_contrast_d7',
    # Multi-level wavelet HH
    'wavelet_hh_l1', 'wavelet_hh_l2', 'wavelet_hh_l3',
    # Multi-radius LBP entropy
    'lbp_entropy_r1', 'lbp_entropy_r2',
    # FFT radial slope
    'fft_radial_slope',
    # Histogram shape
    'hist_bimodality', 'hist_peak_valley',
    # Chroma
    'chroma_std_ratio',
]


def extract_screenshot_features(image_path: str) -> np.ndarray:
    """
    Extract the 15-feature screenshot vector from an image.

    Args:
        image_path: Path to image file.

    Returns:
        np.ndarray of shape (15,), float32.
        Returns zero vector on error or too-small images.
    """
    zero = np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    try:
        img = cv2.imread(image_path)
        if img is None:
            return zero
        h, w = img.shape[:2]
        if min(h, w) < 64:
            return zero

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Multi-scale GLCM
        h1, c1 = _glcm_stats(gray, distance=1)
        h3, c3 = _glcm_stats(gray, distance=3)
        h7, c7 = _glcm_stats(gray, distance=7)

        # Multi-level wavelet
        wv_levels = _haar_hh_energy(gray, levels=3)
        wv_l1     = wv_levels[0] if len(wv_levels) > 0 else 0.0
        wv_l2     = wv_levels[1] if len(wv_levels) > 1 else 0.0
        wv_l3     = wv_levels[2] if len(wv_levels) > 2 else 0.0

        # Multi-radius LBP
        lbp_r1 = _lbp_entropy(gray, radius=1)
        lbp_r2 = _lbp_entropy(gray, radius=2)

        # FFT radial slope
        fft_slope = _fft_radial_slope(gray)

        # Histogram shape
        bim, pv = _histogram_shape(gray)

        # Chroma
        chroma = _chroma_std_ratio(img)

        vec = np.array([
            h1, h3, h7,
            c1, c3, c7,
            wv_l1, wv_l2, wv_l3,
            lbp_r1, lbp_r2,
            fft_slope,
            bim, pv,
            chroma,
        ], dtype=np.float32)

        # Sanitise
        vec = np.where(np.isfinite(vec), vec, 0.0)
        return vec

    except Exception:
        return zero


def extract_screenshot_features_dict(image_path: str) -> Dict[str, float]:
    """Dict version of extract_screenshot_features."""
    vec = extract_screenshot_features(image_path)
    return {name: float(v) for name, v in zip(FEATURE_NAMES, vec)}


# ──────────────────────────────────────────────────────────────────────────────
# ScreenshotClassifier
# ──────────────────────────────────────────────────────────────────────────────

class ScreenshotClassifier:
    """
    Dedicated SVM classifier for AI-screenshot vs real-screenshot detection.

    Trained only on data from:
      - data/ai_generated_screenshots/  (label 1 = AI)
      - data/screenshots/               (label 0 = Real)

    Uses 15 screenshot-specific features at multiple scales so the model
    is not confused by missing ELA/DCT/metadata signals (which are always
    absent for screenshots regardless of origin).
    """

    MODEL_PATH = 'models/screenshot_classifier.pkl'

    def __init__(self) -> None:
        # RBF SVM with lower C than main model — fewer features, less data,
        # need stronger regularisation to avoid overfitting
        self.svm     = SVC(kernel='rbf', probability=True, C=2.0, gamma='scale', class_weight='balanced')
        self.scaler  = StandardScaler()
        self.trained = False

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self,
              ai_ss_dir:   str,
              real_ss_dir: str,
              extra_ai_dirs:   Optional[List[str]] = None,
              extra_real_dirs: Optional[List[str]] = None,
              verbose:     bool = True) -> Dict[str, Any]:
        """
        Train on screenshot-specific data.

        Args:
            ai_ss_dir:       Directory of AI-generated screenshots (label 1).
            real_ss_dir:     Directory of real screenshots         (label 0).
            extra_ai_dirs:   Additional AI screenshot directories.
            extra_real_dirs: Additional real screenshot directories.
            verbose:         Print progress.

        Returns:
            Training results dict.
        """
        img_exts = {'.jpg', '.jpeg', '.png', '.webp'}

        def _list_images(d: str) -> List[str]:
            return [os.path.join(d, f) for f in os.listdir(d)
                    if os.path.splitext(f)[1].lower() in img_exts]

        ai_paths   = _list_images(ai_ss_dir)
        real_paths = _list_images(real_ss_dir)

        for d in (extra_ai_dirs or []):
            if os.path.isdir(d):
                extra = _list_images(d)
                if verbose:
                    print(f"  + {len(extra)} AI screenshots from {d}")
                ai_paths.extend(extra)

        for d in (extra_real_dirs or []):
            if os.path.isdir(d):
                extra = _list_images(d)
                if verbose:
                    print(f"  + {len(extra)} real screenshots from {d}")
                real_paths.extend(extra)

        if verbose:
            print(f"Screenshot classifier training data:")
            print(f"  AI screenshots:   {len(ai_paths)}")
            print(f"  Real screenshots: {len(real_paths)}")
            print(f"  Total:            {len(ai_paths) + len(real_paths)}")

        paths  = ai_paths + real_paths
        labels = np.array([1] * len(ai_paths) + [0] * len(real_paths))

        if len(paths) < 10:
            raise ValueError("Need at least 10 screenshot images to train.")

        # Extract features
        if verbose:
            print("\nExtracting 15 screenshot features per image...")

        X = np.stack([extract_screenshot_features(p) for p in paths], axis=0)

        # Scale
        X_scaled = self.scaler.fit_transform(X)

        # Cross-validation (leave-one-out for small datasets)
        n_folds = min(5, len(paths) // 4)
        if verbose:
            print(f"\nRunning {n_folds}-fold CV...")

        cv_scores = cross_val_score(
            SVC(kernel='rbf', probability=True, C=2.0, gamma='scale'),
            X_scaled, labels, cv=n_folds
        )

        if verbose:
            print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        # Train on full set
        self.svm.fit(X_scaled, labels)
        self.trained = True

        train_acc = accuracy_score(labels, self.svm.predict(X_scaled))
        if verbose:
            print(f"  Train Accuracy: {train_acc:.3f}")

        return {
            'cv_accuracy_mean': float(cv_scores.mean()),
            'cv_accuracy_std':  float(cv_scores.std()),
            'train_accuracy':   float(train_acc),
            'n_ai':             len(ai_paths),
            'n_real':           len(real_paths),
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, image_path: str) -> Dict[str, Any]:
        """
        Classify a screenshot as AI-generated or Real.

        Args:
            image_path: Path to the image file.

        Returns:
            Dict with: label, confidence, scores, method, explanation.
        """
        if not self.trained:
            raise RuntimeError("Screenshot classifier not trained. Run train() first.")

        features        = extract_screenshot_features(image_path).reshape(1, -1)
        features_scaled = self.scaler.transform(features)

        proba_ai = self.svm.predict_proba(features_scaled)[0][1]
        prediction = 1 if proba_ai >= 0.55 else 0
        probabilities = self.svm.predict_proba(features_scaled)[0]

        label      = "AI-Generated" if prediction == 1 else "Real"
        confidence = float(max(probabilities))

        # Build per-feature explanation dict
        feat_dict = extract_screenshot_features_dict(image_path)

        return {
            'label':       label,
            'confidence':  confidence,
            'method':      'screenshot_svm',
            'scores':      {'screenshot_svm_score': float(probabilities[1])},
            'features':    feat_dict,
            'explanation': (
                f"Screenshot classifier ({label}, {confidence:.0%} confidence). "
                f"GLCM contrast@d1={feat_dict['glcm_contrast_d1']:.2f}, "
                f"wavelet_HH_L1={feat_dict['wavelet_hh_l1']:.2f}, "
                f"LBP_entropy_r1={feat_dict['lbp_entropy_r1']:.3f}, "
                f"FFT_slope={feat_dict['fft_radial_slope']:.3f}, "
                f"hist_bimodality={feat_dict['hist_bimodality']:.3f}."
            ),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = MODEL_PATH) -> str:
        """Save to disk."""
        if not self.trained:
            raise RuntimeError("Cannot save: model not trained.")
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        joblib.dump({'svm': self.svm, 'scaler': self.scaler,
                     'feature_names': FEATURE_NAMES}, path)
        print(f"Screenshot classifier saved to {path}")
        return path

    def load(self, path: str = MODEL_PATH) -> None:
        """Load from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Screenshot model not found: {path}")
        data         = joblib.load(path)
        self.svm     = data['svm']
        self.scaler  = data['scaler']
        self.trained = True
        print(f"Screenshot classifier loaded from {path}")

    @property
    def is_trained(self) -> bool:
        return self.trained

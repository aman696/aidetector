"""
screenshot_image_analyzer.py — Screenshot Image Forensics Features

Implements 10 classical features from the recaptured/screenshot image detection
literature that help distinguish AI-generated screenshots from real screenshots.

Research basis:
  - Ng et al. (Imperial College) — GLCM + LBP texture for recapture detection
  - Thongkamwitoon et al. — FFT periodicity (Moiré proxy) for recapture
  - Tan et al. 2010 — Wavelet subband statistics for recaptured images
  - Dirik et al. 2007 — Multi-scale wavelet moments for image origin forensics
  - Chroma subsampling forensics (Erlangen-Nuremberg) — inter-channel analysis

KEY INSIGHT (specific to our problem):
  AI screenshot:   AI-PNG → rendered on sRGB display → OS screenshot → PNG/JPEG
  Real screenshot: Real web/app UI → OS screenshot → PNG/JPEG

Observable differences:
  • Real UIs have fonts, icons, borders → strong periodic/diagonal structures in FFT
  • Real UIs have high local texture diversity (text, icons, backgrounds in close proximity)
  • AI images have smooth, uniform regions → low GLCM contrast, low LBP entropy
  • AI color regions are flat → low chroma channel variance relative to luminance
  • UI elements create step-function intensity transitions, AI has smooth gradients

Feature vector (10 features, all floats, appended after the 54 base features
and before the 15 RIGID drift features):
  0. fft_periodic_score     — prominence of periodic peaks in FFT mid-freq band
  1. fft_peak_to_bg_ratio   — mean of top-N FFT peaks / median background level
  2. glcm_homogeneity       — GLCM: local pair uniformity (high for smooth AI)
  3. glcm_contrast          — GLCM: local contrast (low for smooth AI)
  4. glcm_energy            — GLCM: energy = sum of squared entries
  5. lbp_entropy            — LBP histogram entropy (low for uniform AI regions)
  6. wavelet_hh_energy       — DWT HH subband energy (diagonal high-freq, low for AI)
  7. wavelet_ratio_hh_ll     — HH/LL subband energy ratio (AI images have low ratio)
  8. chroma_std_ratio        — std(Cb) / std(Cr)  deviation from 1.0 (AI is flatter)
  9. tone_step_density       — fraction of pixels on steep histogram slopes (UI has steps)
"""

import numpy as np
import cv2
from typing import Dict


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

_ZERO_FEATURES: Dict[str, float] = {
    'fft_periodic_score':   0.0,
    'fft_peak_to_bg_ratio': 0.0,
    'glcm_homogeneity':     0.0,
    'glcm_contrast':        0.0,
    'glcm_energy':          0.0,
    'lbp_entropy':          0.0,
    'wavelet_hh_energy':    0.0,
    'wavelet_ratio_hh_ll':  0.0,
    'chroma_std_ratio':     0.0,
    'tone_step_density':    0.0,
}


# ──────────────────────────────────────────────────────────────────────────────
# 1. FFT Periodicity — Thongkamwitoon et al. / Moiré proxy
# ──────────────────────────────────────────────────────────────────────────────

def _fft_periodicity(gray: np.ndarray) -> tuple[float, float]:
    """
    Detect periodic peaks in the mid-frequency band of the 2D FFT magnitude.

    Real screenshots of UI content (text, icon grids, pixel-art) produce
    periodic spectral peaks because of their regular sub-structures.
    AI-generated image screenshots have smooth, aperiodic frequency content.

    We look at the mid-frequency annular ring (10%–60% of max radius),
    compute the top-N peak-to-background ratio as the primary score.

    Returns:
        (periodic_score, peak_to_bg_ratio) — both in [0, ~inf], higher = more periodic.
    """
    h, w = gray.shape
    # Center-crop to make square and reduce FFT size for speed
    side = min(h, w, 512)
    cy, cx = h // 2, w // 2
    crop = gray[cy - side//2: cy + side//2, cx - side//2: cx + side//2]

    fft = np.fft.fft2(crop.astype(np.float32))
    fft_shift = np.fft.fftshift(fft)
    mag = np.abs(fft_shift)

    # Zero out DC (center)
    cy2, cx2 = side // 2, side // 2
    mag[cy2 - 3: cy2 + 4, cx2 - 3: cx2 + 4] = 0

    # Build radius map
    ys, xs = np.ogrid[-cy2: side - cy2, -cx2: side - cx2]
    r = np.sqrt(xs**2 + ys**2)
    max_r = np.sqrt(cx2**2 + cy2**2)

    # Mid-frequency annulus: 10% – 60% of max radius
    mid_mask = (r >= 0.10 * max_r) & (r <= 0.60 * max_r)
    mid_vals = mag[mid_mask]

    if mid_vals.size == 0:
        return 0.0, 0.0

    # Top-1% peaks vs median background
    n_top = max(1, int(0.01 * mid_vals.size))
    sorted_vals = np.sort(mid_vals)
    top_mean = sorted_vals[-n_top:].mean()
    bg_median = np.median(sorted_vals)

    peak_to_bg = float(top_mean / (bg_median + 1e-8))

    # Periodic score: std of the angular distribution of the top peaks
    # If there are sharp directional peaks → high std → periodic
    top_indices  = np.where(mid_vals >= sorted_vals[-n_top])
    # Flatten back to image coordinates
    flat_idx = np.flatnonzero(mid_mask)
    if len(flat_idx) == 0 or len(top_indices[0]) == 0:
        return 0.0, float(peak_to_bg)

    top_flat    = flat_idx[top_indices[0]]
    top_rows    = top_flat // side - cy2
    top_cols    = top_flat  % side - cx2
    angles      = np.arctan2(top_rows.astype(float), top_cols.astype(float))
    angle_std   = float(np.std(angles))          # high if peaks cluster → periodic
    periodic    = float(np.clip(angle_std, 0, np.pi))

    return periodic, float(peak_to_bg)


# ──────────────────────────────────────────────────────────────────────────────
# 2. GLCM — Ng et al. (Imperial College) texture features
# ──────────────────────────────────────────────────────────────────────────────

def _glcm_features(gray: np.ndarray,
                   distances=(1, 3),
                   angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)
                   ) -> tuple[float, float, float]:
    """
    Compute GLCM (Gray-Level Co-occurrence Matrix) and extract:
      - homogeneity (high for smooth/uniform regions, i.e. AI images)
      - contrast    (low for smooth regions)
      - energy      (high for highly uniform repetitive textures)

    Uses a reduced quantisation (32 gray levels) for efficiency.
    Averages across multiple distances and angles for robustness.

    Reference: Ng et al. — GLCM texture for screen-recapture detection.
    """
    # Quantise to 32 levels
    q = (gray // 8).astype(np.int32)
    levels = 32

    homo_vals, contr_vals, engy_vals = [], [], []

    for d in distances:
        for angle in angles:
            dx = int(round(d * np.cos(angle)))
            dy = int(round(d * np.sin(angle)))

            # Shift
            if dx == 0 and dy == 0:
                continue

            rows, cols = q.shape
            # Crop to valid region
            r0, r1 = max(0, -dy), min(rows, rows - dy)
            c0, c1 = max(0, -dx), min(cols, cols - dx)

            pixel1 = q[r0:r1, c0:c1]
            pixel2 = q[r0 + dy: r1 + dy, c0 + dx: c1 + dx]

            if pixel1.size == 0:
                continue

            # Build co-occurrence matrix
            glcm = np.zeros((levels, levels), dtype=np.float64)
            np.add.at(glcm, (pixel1.ravel(), pixel2.ravel()), 1)
            # Symmetrise
            glcm = glcm + glcm.T
            total = glcm.sum()
            if total < 1:
                continue
            glcm /= total

            i_idx, j_idx = np.meshgrid(np.arange(levels), np.arange(levels), indexing='ij')

            homo_vals.append(float((glcm / (1 + np.abs(i_idx - j_idx))).sum()))
            contr_vals.append(float((glcm * (i_idx - j_idx)**2).sum()))
            engy_vals.append(float((glcm**2).sum()))

    if not homo_vals:
        return 0.5, 0.5, 0.5

    return float(np.mean(homo_vals)), float(np.mean(contr_vals)), float(np.mean(engy_vals))


# ──────────────────────────────────────────────────────────────────────────────
# 3. LBP Entropy — multi-scale texture diversity
# ──────────────────────────────────────────────────────────────────────────────

def _lbp_entropy(gray: np.ndarray, radius: int = 1) -> float:
    """
    Compute LBP (Local Binary Pattern) histogram entropy.

    Real screenshots contain diverse local textures (fonts, icons, backgrounds
    in close proximity) → high histogram entropy.
    AI image screenshots have uniform smooth regions → low entropy.

    Uses the basic LBP (8 neighbours, radius 1) for speed.
    Returns Shannon entropy of the 256-bin LBP histogram.

    Reference: Ng et al., various recapture detection papers.
    """
    h, w = gray.shape
    lbp = np.zeros((h - 2*radius, w - 2*radius), dtype=np.uint8)

    # Extract the 8 neighbours at radius=1 (3x3 window)
    center = gray[radius: h - radius, radius: w - radius].astype(np.int16)

    offsets = [(-1, -1), (-1, 0), (-1, 1),
               ( 0,  1), ( 1,  1), ( 1,  0),
               ( 1, -1), ( 0, -1)]

    for bit, (dy, dx) in enumerate(offsets):
        r0, r1 = radius + dy, h - radius + dy
        c0, c1 = radius + dx, w - radius + dx
        neighbour = gray[r0:r1, c0:c1].astype(np.int16)
        lbp |= ((neighbour >= center).astype(np.uint8) << bit)

    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist    = hist.astype(np.float64)
    hist   /= (hist.sum() + 1e-10)
    # Shannon entropy normalised to [0, 1]
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return float(entropy / 8.0)   # max entropy = 8 bits


# ──────────────────────────────────────────────────────────────────────────────
# 4. Wavelet HH Subband — Tan et al. 2010
# ──────────────────────────────────────────────────────────────────────────────

def _wavelet_features(gray: np.ndarray) -> tuple[float, float]:
    """
    Apply one-level Haar DWT and extract energy from the HH (diagonal) subband.

    Real screenshots have strong diagonal structures from fonts, icon edges,
    and borders → high HH energy. AI-generated images are smoother.

    HH/LL ratio separates AI (low HH, high LL) from real UI content (higher HH).

    Reference: Tan et al. 2010 — wavelet moment analysis for recaptured detection.
    """
    img = gray.astype(np.float32)

    # Haar DWT: LL, LH, HL, HH via simple row/col averaging and differencing
    # Downsample by 2
    h, w = img.shape
    # Make even
    img = img[:h - (h % 2), :w - (w % 2)]
    h, w = img.shape

    rows_even = img[0::2, :]
    rows_odd  = img[1::2, :]

    low_row  = (rows_even + rows_odd)  / 2.0     # low-pass rows
    high_row = (rows_even - rows_odd)  / 2.0     # high-pass rows

    cols_even = lambda m: m[:, 0::2]
    cols_odd  = lambda m: m[:, 1::2]

    LL = (cols_even(low_row)  + cols_odd(low_row))  / 2.0
    # LH = (cols_even(low_row)  - cols_odd(low_row))  / 2.0
    # HL = (cols_even(high_row) + cols_odd(high_row)) / 2.0
    HH = (cols_even(high_row) - cols_odd(high_row)) / 2.0

    ll_energy = float(np.mean(LL**2))
    hh_energy = float(np.mean(HH**2))

    ratio = float(hh_energy / (ll_energy + 1e-8))
    # Normalise HH energy by image size
    norm_hh = float(np.log1p(hh_energy))
    return norm_hh, ratio


# ──────────────────────────────────────────────────────────────────────────────
# 5. Chroma Channel Std Ratio — chromaticity forensics
# ──────────────────────────────────────────────────────────────────────────────

def _chroma_std_ratio(img_bgr: np.ndarray) -> float:
    """
    Measure the imbalance between Cb and Cr channel standard deviations after
    converting to YCbCr.

    AI-generated images tend to have flat, correlated chroma channels
    (the generator produces balanced colour). Real screenshots contain
    saturated icon colours + desaturated text → uneven Cb/Cr distribution.

    Returns: |std(Cb) / (std(Cr) + ε) - 1|  — 0 = perfectly balanced, >0 = imbalanced.

    Reference: chroma subsampling / chromaticity forensics (Univ. Erlangen-Nuremberg).
    """
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    cb = ycrcb[:, :, 2]   # OpenCV YCrCb: channel 1=Cr, 2=Cb
    cr = ycrcb[:, :, 1]

    std_cb = float(np.std(cb))
    std_cr = float(np.std(cr))

    ratio = abs(std_cb / (std_cr + 1e-8) - 1.0)
    return float(np.clip(ratio, 0, 5))


# ──────────────────────────────────────────────────────────────────────────────
# 6. Tone Step Density — UI step-function intensity analysis
# ──────────────────────────────────────────────────────────────────────────────

def _tone_step_density(gray: np.ndarray, bins: int = 64) -> float:
    """
    Estimate the fraction of the intensity histogram that sits on steep slopes.

    UI content (text on background, buttons, icons) creates sharp histogram
    peaks at specific intensity values separated by valleys — a 'stepped'
    pattern. AI-generated image screenshots have smoother histograms because
    diffusion models produce continuous gradient distributions.

    We compute the fraction of histogram bins that have gradient
    |Δhist| > threshold (relative to local mean) as the step density.

    Returns: fraction in [0, 1], higher = more step-like (more like real UI).
    """
    hist, _ = np.histogram(gray.ravel(), bins=bins, range=(0, 256))
    hist    = hist.astype(np.float64)
    hist   /= (hist.max() + 1e-10)         # normalise peak to 1.0

    grad    = np.abs(np.diff(hist))        # first-order difference
    local_mean = np.convolve(grad, np.ones(5) / 5, mode='same')
    threshold   = 2.0 * local_mean + 0.05  # adaptive threshold

    steep   = (grad > threshold).sum()
    density = float(steep / len(grad))
    return density


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def extract_screenshot_image_features(image_path: str) -> Dict[str, float]:
    """
    Extract 10 screenshot-forensics features from an image.

    Features are designed to separate AI-generated image screenshots from
    real content screenshots (UI, web pages, apps).

    Args:
        image_path: Absolute path to image file (.jpg / .png / .webp).

    Returns:
        Dict with 10 float features (all finite, no NaN/Inf).
        Returns _ZERO_FEATURES on any error.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return dict(_ZERO_FEATURES)

        h, w = img.shape[:2]
        if min(h, w) < 64:
            return dict(_ZERO_FEATURES)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1 & 2: FFT periodicity
        periodic_score, peak_to_bg = _fft_periodicity(gray)

        # 3, 4, 5: GLCM
        glcm_homo, glcm_contr, glcm_engy = _glcm_features(gray)

        # 6: LBP entropy
        lbp_ent = _lbp_entropy(gray)

        # 7 & 8: Wavelet HH
        wv_hh, wv_ratio = _wavelet_features(gray)

        # 9: Chroma std ratio
        chroma_ratio = _chroma_std_ratio(img)

        # 10: Tone step density
        step_dens = _tone_step_density(gray)

        features = {
            'fft_periodic_score':   float(np.clip(periodic_score, 0, 10)),
            'fft_peak_to_bg_ratio': float(np.clip(peak_to_bg,    0, 100)),
            'glcm_homogeneity':     float(np.clip(glcm_homo,     0, 1)),
            'glcm_contrast':        float(np.clip(glcm_contr,    0, 1000)),
            'glcm_energy':          float(np.clip(glcm_engy,     0, 1)),
            'lbp_entropy':          float(np.clip(lbp_ent,       0, 1)),
            'wavelet_hh_energy':    float(np.clip(wv_hh,         0, 20)),
            'wavelet_ratio_hh_ll':  float(np.clip(wv_ratio,      0, 1)),
            'chroma_std_ratio':     float(np.clip(chroma_ratio,  0, 5)),
            'tone_step_density':    float(np.clip(step_dens,     0, 1)),
        }

        # Sanity check: replace any NaN/Inf that slipped through
        for k, v in features.items():
            if not np.isfinite(v):
                features[k] = 0.0

        return features

    except Exception:
        return dict(_ZERO_FEATURES)


def screenshot_image_score(image_path: str) -> float:
    """
    Single aggregate score: probability image is a REAL screenshot (not AI).
    Returns value in [0, 1] where higher = more likely real UI screenshot.

    Weights calibrated against observed data (21 AI-SS vs 14 real-SS):

    Feature            AI avg   Real avg   Dir     Used?
    ─────────────────  ──────   ────────   ─────   ─────
    glcm_energy         0.032    0.403    ↑Real  ✅ strongest
    wavelet_hh_energy   1.613    2.311    ↑Real  ✅ strong
    glcm_contrast       7.10    17.65     ↑Real  ✅ strong
    lbp_entropy         0.719    0.194    ↓Real  ✅ inverted (AI-SS are large uniform images)
    fft_periodic_score  1.734    1.518    ↓Real  ✅ AI has more FFT periodicity (diffusion grid)
    chroma_std_ratio    0.257    0.338    ↑Real  ⚠ weak
    glcm_homogeneity   same direction → dropped
    fft_peak_to_bg    ↓Real strong in raw but correlated with periodic → kept light
    wavelet_ratio      tiny  → dropped
    tone_step_density  tiny  → dropped
    """
    f = extract_screenshot_image_features(image_path)

    # Normalise each feature to ~ [0, 1] range
    glcm_e_norm  = float(np.clip(f['glcm_energy'], 0, 1))                     # already 0-1
    wv_hh_norm   = float(np.clip(f['wavelet_hh_energy'] / 5.0, 0, 1))         # typical 0-5
    glcm_c_norm  = float(np.clip(f['glcm_contrast'] / 50.0, 0, 1))            # typical 0-50
    lbp_inv      = float(1.0 - np.clip(f['lbp_entropy'], 0, 1))               # inverted: high LBP=AI
    fft_p_inv    = float(1.0 - np.clip(f['fft_periodic_score'] / 3.0, 0, 1))  # lower=real
    chroma_norm  = float(np.clip(f['chroma_std_ratio'] / 2.0, 0, 1))

    score = (
        0.30 * glcm_e_norm  +   # strongest signal: real UI has big GLCM energy
        0.25 * wv_hh_norm   +   # real UI has strong diagonal wavelet energy
        0.20 * glcm_c_norm  +   # real UI has high local contrast
        0.15 * lbp_inv      +   # AI-SS have flat uniform regions → high LBP, so invert
        0.07 * fft_p_inv    +   # AI-SS have slightly more spectral periodicity
        0.03 * chroma_norm      # real-SS have slightly more chroma imbalance
    )

    return float(np.clip(score, 0.0, 1.0))


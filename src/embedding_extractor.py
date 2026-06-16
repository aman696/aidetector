"""
DINOv2 embedding extractor with RIGID-style noise-robustness drift.

Produces, per image, a 770-dimensional vector:
    - 768 dims: the DINOv2 ViT-B/14 image embedding (pooled CLS token).
    - 2 dims:  RIGID drift = 1 - cosine(embed(clean), embed(clean + noise))
               at two noise levels (sigma = 2/255 and 6/255 in [0, 1] image
               space). Real images are more robust to tiny perturbations, so
               their drift is smaller; larger drift reads as more AI-like.

The repository must keep working
without torch/timm, so those are imported lazily inside DinoEmbedder; this
module is importable (and the classical pipeline runnable) without them.

Math notes that matter for correctness:
    - The RIGID noise is added in [0, 1] image space BEFORE ImageNet
      normalization, then clipped to [0, 1]. Adding it after normalization
      would scale the effective sigma by 1/std (~1/0.225) per channel and
      break the "2 and 6 gray levels out of 255" interpretation.
    - The noise is seeded per image (md5 of the crop pixels, or a caller-
      supplied seed) so the cached drift features are reproducible — unlike
      the legacy classical drift in classifier.py, which is unseeded.
    - Cosine similarity is computed in float64 even though embeddings are
      produced in fp16 on GPU, to avoid normalization precision loss.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

from .patchcraft_analyzer import compute_high_pass, compute_patch_ldiv

ImageInput = Union[str, Path, np.ndarray]

DEFAULT_MODEL = "vit_base_patch14_dinov2.lvd142m"
CROP_SIZE = 448          # texture-rich crop side; 2x the model input
INPUT_SIZE = 224         # DINOv2 ViT-B/14 input (16x16 grid of 14px patches)
GRID_PATCH = 32          # patch size for the texture-richness scoring grid
DRIFT_SIGMAS = (2.0 / 255.0, 6.0 / 255.0)
OUT_DIM = 768 + len(DRIFT_SIGMAS)


# --------------------------------------------------------------------------- #
# Image loading and texture-rich cropping (no torch needed)
# --------------------------------------------------------------------------- #

def _composite_on_white(arr: np.ndarray) -> np.ndarray:
    """Alpha-composite an HxWx4 uint8 array onto a white background."""
    rgb = arr[..., :3].astype(np.float32)
    alpha = arr[..., 3:4].astype(np.float32) / 255.0
    out = rgb * alpha + 255.0 * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_rgb(image: ImageInput) -> np.ndarray:
    """
    Load an image as an HxWx3 uint8 RGB array.

    RGBA / palette / grayscale inputs are converted to RGB; transparency is
    composited onto white, matching scripts/build_derived.py so the embedder
    and the derived-data pipeline agree on what a transparent pixel "is".
    """
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 4:
            return _composite_on_white(arr)
        return np.ascontiguousarray(arr[..., :3]).astype(np.uint8)

    from PIL import Image
    img = Image.open(str(image))
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _ldiv_grid(high_pass: np.ndarray, patch: int) -> Optional[np.ndarray]:
    """Per-patch l_div as a 2D (ph, pw) grid (None if the image is too small)."""
    ph, pw = high_pass.shape[0] // patch, high_pass.shape[1] // patch
    if ph == 0 or pw == 0:
        return None
    flat = compute_patch_ldiv(high_pass, patch)
    if flat.size != ph * pw:
        return None
    return flat.reshape(ph, pw)


def _best_window(grid: np.ndarray, win: int) -> tuple[int, int]:
    """
    Top-left (row, col) in patch units of the win x win window with the
    greatest summed l_div, via an integral image. Falls back to centered
    if the grid is smaller than the window.
    """
    ph, pw = grid.shape
    if ph < win or pw < win:
        return max(0, (ph - win) // 2), max(0, (pw - win) // 2)
    ii = grid.cumsum(0).cumsum(1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    out_h, out_w = ph - win + 1, pw - win + 1
    window_sums = (
        ii[win:win + out_h, win:win + out_w]
        - ii[0:out_h, win:win + out_w]
        - ii[win:win + out_h, 0:out_w]
        + ii[0:out_h, 0:out_w]
    )
    idx = int(np.argmax(window_sums))
    return idx // out_w, idx % out_w


def texture_rich_crop(rgb: np.ndarray) -> np.ndarray:
    """
    Return a 224x224x3 uint8 RGB crop favouring high-texture regions.

    Large images (min side >= 448): pick the 448x448 window richest in
    texture (highest summed l_div), then area-downscale 448 -> 224 (a clean
    2x integer downscale). Small images: resize the shorter side to 256 and
    center-crop 224 (the standard ViT eval path).
    """
    import cv2

    h, w = rgb.shape[:2]
    if min(h, w) >= CROP_SIZE:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        grid = _ldiv_grid(compute_high_pass(gray), GRID_PATCH)
        win = CROP_SIZE // GRID_PATCH  # 14 patches = 448 px
        if grid is None:
            y0 = (h - CROP_SIZE) // 2
            x0 = (w - CROP_SIZE) // 2
        else:
            top, left = _best_window(grid, win)
            y0, x0 = top * GRID_PATCH, left * GRID_PATCH
        crop = rgb[y0:y0 + CROP_SIZE, x0:x0 + CROP_SIZE]
        return cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE),
                          interpolation=cv2.INTER_AREA)

    scale = 256.0 / min(h, w)
    new_h, new_w = round(h * scale), round(w * scale)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)
    y0 = max(0, (new_h - INPUT_SIZE) // 2)
    x0 = max(0, (new_w - INPUT_SIZE) // 2)
    return resized[y0:y0 + INPUT_SIZE, x0:x0 + INPUT_SIZE]


def _seed_from_array(arr: np.ndarray) -> int:
    """Deterministic 63-bit seed from pixel bytes (md5; stable across processes)."""
    digest = hashlib.md5(np.ascontiguousarray(arr).tobytes()).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63)


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity in float64 (precision-safe over fp16 inputs)."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    num = np.sum(a64 * b64, axis=1)
    den = np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1) + 1e-12
    return (num / den).astype(np.float32)


# --------------------------------------------------------------------------- #
# Embedder (torch/timm, imported lazily)
# --------------------------------------------------------------------------- #

class DinoEmbedder:
    """
    Frozen DINOv2 ViT-B/14 feature extractor producing clean embeddings and
    RIGID drift. Construct once and reuse (loading weights is the expensive
    step). torch/timm are imported here, so importing this module costs
    nothing when they are absent.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        half: Optional[bool] = None,
    ) -> None:
        import timm
        import torch

        self.torch = torch
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # fp16 only buys throughput/memory on GPU; fp16 on CPU is slow.
        self.half = self.device.startswith("cuda") if half is None else half

        model = timm.create_model(
            model_name, pretrained=True, num_classes=0, img_size=INPUT_SIZE
        )
        model.eval()
        cfg = timm.data.resolve_model_data_config(model)
        self.mean = np.array(cfg["mean"], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(cfg["std"], dtype=np.float32).reshape(3, 1, 1)
        self.embed_dim = int(model.num_features)
        self.out_dim = self.embed_dim + len(DRIFT_SIGMAS)

        model.to(self.device)
        if self.half:
            model.half()
        self.model = model

    # -- internals ---------------------------------------------------------- #

    def _normalized_tensor(self, x01):
        """[0,1] CHW float32 torch tensor -> ImageNet-normalized tensor."""
        torch = self.torch
        mean = torch.from_numpy(self.mean)
        std = torch.from_numpy(self.std)
        return (x01 - mean) / std

    def _forward(self, tensors: list, batch_size: int) -> np.ndarray:
        torch = self.torch
        outs = []
        with torch.inference_mode():
            for k in range(0, len(tensors), batch_size):
                batch = torch.stack(tensors[k:k + batch_size]).to(self.device)
                if self.half:
                    batch = batch.half()
                feats = self.model(batch)
                outs.append(feats.float().cpu().numpy())
        return np.concatenate(outs, axis=0)

    def _prepare(self, image: ImageInput):
        """Load + texture-crop -> (3,224,224) [0,1] float32 torch tensor + crop."""
        torch = self.torch
        crop = texture_rich_crop(_load_rgb(image))
        chw = np.transpose(crop.astype(np.float32) / 255.0, (2, 0, 1))
        return torch.from_numpy(np.ascontiguousarray(chw)), crop

    # -- public API --------------------------------------------------------- #

    def embed_batch_with_drift(
        self,
        images: Sequence[ImageInput],
        seeds: Optional[Sequence[Optional[int]]] = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Return an (N, 770) array: 768 clean embedding dims + 2 drift dims.

        For each image three forward passes are stacked (clean, noisy@sigma2,
        noisy@sigma6) and run through the model in chunks of batch_size.
        """
        torch = self.torch
        n = len(images)
        tensors: List[object] = []
        for i in range(n):
            x01, crop = self._prepare(images[i])
            seed = None
            if seeds is not None and seeds[i] is not None:
                seed = int(seeds[i])
            if seed is None:
                seed = _seed_from_array(crop)
            gen = torch.Generator().manual_seed(seed % (2 ** 63))

            tensors.append(self._normalized_tensor(x01))  # clean
            for sigma in DRIFT_SIGMAS:
                noise = torch.randn(x01.shape, generator=gen) * float(sigma)
                noisy01 = torch.clamp(x01 + noise, 0.0, 1.0)
                tensors.append(self._normalized_tensor(noisy01))

        embs = self._forward(tensors, batch_size)
        n_per = 1 + len(DRIFT_SIGMAS)
        embs = embs.reshape(n, n_per, -1)
        clean = embs[:, 0, :]

        out = np.empty((n, self.out_dim), dtype=np.float32)
        out[:, : self.embed_dim] = clean
        for j in range(len(DRIFT_SIGMAS)):
            out[:, self.embed_dim + j] = 1.0 - _cosine_rows(clean, embs[:, j + 1, :])
        return out

    def embed_with_drift(
        self, image: ImageInput, seed: Optional[int] = None, batch_size: int = 32
    ) -> np.ndarray:
        """Single-image convenience wrapper returning a (770,) vector."""
        s = None if seed is None else [seed]
        return self.embed_batch_with_drift([image], s, batch_size)[0]

    def embed(
        self, images: Sequence[ImageInput], batch_size: int = 32
    ) -> np.ndarray:
        """Clean embeddings only, (N, 768) — for embedding-only ablations."""
        tensors = [self._normalized_tensor(self._prepare(im)[0]) for im in images]
        return self._forward(tensors, batch_size)

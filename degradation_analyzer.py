"""
Degradation Analyzer Module
============================
Estimates image degradation severity using no-reference quality metrics.
Produces a degradation score ∈ [0, 1] that drives adaptive timestep selection
and wavelet fusion weighting in the advanced OSDFace pipeline.

Metrics:
    1. Laplacian Sharpness — low variance indicates blur
    2. Local Noise Estimation — high-frequency patch-level noise
    3. Signal Quality — contrast and intensity distribution
"""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class DegradationAnalyzer:
    """Analyze image degradation level for adaptive restoration control."""

    def __init__(self, t_min: int = 200, t_max: int = 600):
        """
        Args:
            t_min: Minimum diffusion timestep (for clean images).
            t_max: Maximum diffusion timestep (for heavily degraded images).
        """
        self.t_min = t_min
        self.t_max = t_max

    # --------------------------------------------------------------------- #
    #  Individual quality metrics (operate on numpy HWC uint8 or float)
    # --------------------------------------------------------------------- #

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        """Convert HWC image to single-channel grayscale float64."""
        if img.ndim == 3 and img.shape[2] == 3:
            return np.dot(img[..., :3].astype(np.float64), [0.2989, 0.5870, 0.1140])
        return img.astype(np.float64)

    def _laplacian_sharpness(self, gray: np.ndarray) -> float:
        """
        Variance of discrete Laplacian — proxy for sharpness.
        High value = sharp, low value = blurry.
        Returns a normalized score ∈ [0, 1] where 1 = very blurry.
        """
        # 3×3 Laplacian kernel
        kernel = np.array([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=np.float64)

        from scipy.signal import convolve2d
        laplacian = convolve2d(gray, kernel, mode='same', boundary='symm')
        lap_var = np.var(laplacian)

        # Normalize: empirically, sharp faces have lap_var > 500,
        # severely degraded ones < 50.  Clamp and invert.
        sharp_threshold = 800.0
        blur_score = 1.0 - min(lap_var / sharp_threshold, 1.0)
        return float(blur_score)

    def _noise_estimate(self, gray: np.ndarray) -> float:
        """
        Estimate spatially-varying noise via local standard deviation.
        High global median of local std ⇒ noisy image.
        Returns normalized score ∈ [0, 1] where 1 = very noisy.
        """
        h, w = gray.shape
        patch = 16
        stds = []
        for i in range(0, h - patch + 1, patch):
            for j in range(0, w - patch + 1, patch):
                block = gray[i:i + patch, j:j + patch]
                stds.append(np.std(block))
        if not stds:
            return 0.0
        median_std = float(np.median(stds))
        # Normalize: clean face patches ≈ std 20-30, noisy ≈ 60+
        noise_score = min(median_std / 80.0, 1.0)
        return noise_score

    def _signal_quality(self, gray: np.ndarray) -> float:
        """
        Evaluate overall signal quality via contrast (std / mean).
        Low contrast ⟹ washed out / low quality.
        Returns normalized score ∈ [0, 1] where 1 = poor quality.
        """
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        if mean_val < 1e-6:
            return 1.0
        contrast_ratio = std_val / mean_val
        # Good faces have contrast_ratio ≈ 0.3–0.5; poor < 0.15
        quality_score = 1.0 - min(contrast_ratio / 0.5, 1.0)
        return float(quality_score)

    # --------------------------------------------------------------------- #
    #  Aggregate analysis
    # --------------------------------------------------------------------- #

    def _compute_degradation_score(self, img_np: np.ndarray) -> dict:
        """
        Compute individual metrics and aggregate degradation score.

        Args:
            img_np: HWC uint8 numpy image (RGB).

        Returns:
            dict with keys: blur_score, noise_score, quality_score,
                            degradation_score, suggested_timestep, fusion_alpha
        """
        gray = self._to_gray(img_np)

        blur = self._laplacian_sharpness(gray)
        noise = self._noise_estimate(gray)
        quality = self._signal_quality(gray)

        # Weighted aggregate — blur is most important for face restoration
        degradation_score = float(np.clip(
            0.45 * blur + 0.30 * noise + 0.25 * quality, 0.0, 1.0
        ))

        timestep = self.score_to_timestep(degradation_score)
        fusion_alpha = self.score_to_fusion_weight(degradation_score)

        return {
            "blur_score": round(blur, 4),
            "noise_score": round(noise, 4),
            "quality_score": round(quality, 4),
            "degradation_score": round(degradation_score, 4),
            "suggested_timestep": timestep,
            "fusion_alpha": round(fusion_alpha, 4),
        }

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #

    def analyze(self, image_tensor: torch.Tensor) -> dict:
        """
        Analyze a batch tensor or single image tensor.

        Args:
            image_tensor: (C, H, W) or (1, C, H, W) float tensor in [0, 1].

        Returns:
            dict with degradation metrics.
        """
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        img_np = (image_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return self._compute_degradation_score(img_np)

    def analyze_from_path(self, image_path: str) -> dict:
        """Convenience: analyze directly from an image file path."""
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        return self._compute_degradation_score(img_np)

    def score_to_timestep(self, score: float, t_min: int = None, t_max: int = None) -> int:
        """
        Map degradation score to diffusion timestep.

        Clean  (score ≈ 0) → t_min (gentle restoration)
        Noisy  (score ≈ 1) → t_max (aggressive restoration)
        """
        t_min = t_min or self.t_min
        t_max = t_max or self.t_max
        t = int(t_min + score * (t_max - t_min))
        return max(t_min, min(t, t_max))

    def score_to_fusion_weight(self, score: float) -> float:
        """
        Map degradation score → wavelet fusion alpha.

        alpha close to 1.0  ⟹ preserve original low-freq (clean image)
        alpha close to 0.0  ⟹ trust restored low-freq   (degraded image)
        """
        alpha = 1.0 - score  # simple linear mapping
        return float(np.clip(alpha, 0.0, 1.0))

"""
Adaptive OSDFace Pipeline
===========================
Consistency-aware, adaptive, and controllable face restoration system.

Three key innovations over the baseline OSDFace inference:
    1. Adaptive Degradation-Aware Timestep Selection
    2. Classifier-Free Guidance (CFG) for Conditioning Control
    3. Wavelet Domain Fusion for Color Preservation

All innovations are inference-only — no retraining required.
"""

import copy
import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.nn.functional as Fun
from torchvision import transforms
import torchvision.transforms.functional as F

from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
from safetensors import safe_open

from utils.vaehook import perfcount
from utils.others import get_x0_from_noise
from utils.wavelet_color_fix import wavelet_decomposition
from lq_embed import vqvae_encoder, TwoLayerConv1x1
from degradation_analyzer import DegradationAnalyzer


class OSDFace_Advanced(nn.Module):
    """
    Advanced OSDFace inference model with:
        - Adaptive degradation-aware timestep selection
        - Classifier-free guidance for identity preservation
        - Wavelet domain fusion for color consistency
    """

    def __init__(self, args, gpu_id, Unet):
        super().__init__()

        self.args = args
        self.device = torch.device(f"cuda:{gpu_id}")

        # ── Scheduler & Core Models ──────────────────────────────────────
        self.noise_scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="scheduler"
        )
        self.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)

        self.vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae"
        )

        if args.merge_lora:
            self.unet = copy.deepcopy(Unet)
        else:
            self.unet = UNet2DConditionModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="unet"
            )

        self.weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

        self.load_ckpt(args.ckpt_path)

        # ── Image Encoder ────────────────────────────────────────────────
        self.img_encoder = vqvae_encoder(args).to(self.device, dtype=self.weight_dtype)
        self.unet.to(self.device, dtype=self.weight_dtype)
        self.vae.to(self.device, dtype=self.weight_dtype)
        self.img_encoder.to(self.device, dtype=self.weight_dtype)

        # ── Baseline fallback timestep (same as original OSDFace) ────────
        self.default_timesteps = 399

        # ── Innovation 1: Adaptive Timestep ──────────────────────────────
        self.enable_adaptive_timestep = not getattr(args, 'disable_adaptive_timestep', False)
        self.degradation_analyzer = DegradationAnalyzer(
            t_min=getattr(args, 't_min', 200),
            t_max=getattr(args, 't_max', 600),
        )

        # ── Innovation 2: Classifier-Free Guidance ───────────────────────
        self.enable_cfg = not getattr(args, 'disable_cfg', False)
        self.guidance_scale = getattr(args, 'guidance_scale', 1.5)

        # ── Innovation 3: Wavelet Fusion ─────────────────────────────────
        self.enable_wavelet_fusion = not getattr(args, 'disable_wavelet_fusion', False)
        self.fusion_strength = getattr(args, 'fusion_strength', None)  # None = adaptive

    def load_ckpt(self, ckpt_path):
        """Load checkpoint weights (same logic as baseline OSDFace_test)."""
        if not self.args.cat_prompt_embedding:
            self.embedding_change = TwoLayerConv1x1(512, 1024)
            self.embedding_change.load_state_dict(
                torch.load(
                    os.path.join(ckpt_path, "embedding_change_weights.pth"),
                    weights_only=False,
                )
            )
            self.embedding_change.to(self.device, dtype=self.weight_dtype)

        if not self.args.merge_lora:
            pipe = StableDiffusionPipeline(
                self.vae, None, None, self.unet,
                self.noise_scheduler, None, None,
            )
            pipe.load_lora_weights(ckpt_path)
            self.unet = pipe.unet

    # -------------------------------------------------------------------- #
    #  Degradation Analysis
    # -------------------------------------------------------------------- #

    def _analyze_degradation(self, lq: torch.Tensor) -> dict:
        """
        Estimate input degradation. Works on the [0, 1] image tensor.

        Args:
            lq: (B, C, H, W) tensor in [-1, 1] range.

        Returns:
            dict with degradation metrics.
        """
        # Convert from [-1, 1] to [0, 1] for the analyzer
        img_01 = (lq[0] * 0.5 + 0.5).clamp(0, 1)
        return self.degradation_analyzer.analyze(img_01)

    # -------------------------------------------------------------------- #
    #  Wavelet Fusion
    # -------------------------------------------------------------------- #

    def _wavelet_fusion(
        self,
        restored: torch.Tensor,
        original: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Adaptive wavelet-domain fusion.

        Blends low-frequency (color/illumination) from original and restored
        images based on alpha, while keeping high-frequency details from the
        restored output.

        Args:
            restored: (B, C, H, W) restored image in [0, 1].
            original: (B, C, H, W) original image in [0, 1].
            alpha: Fusion weight. 1.0 = fully preserve original color,
                   0.0 = fully trust restored color.

        Returns:
            Fused image tensor in [0, 1].
        """
        # Decompose both images
        restored_high, restored_low = wavelet_decomposition(restored)
        original_high, original_low = wavelet_decomposition(original)

        # Blend low-frequency components
        fused_low = alpha * original_low + (1.0 - alpha) * restored_low

        # Combine restored high-freq details with blended low-freq color
        fused = restored_high + fused_low
        return fused.clamp(0.0, 1.0)

    # -------------------------------------------------------------------- #
    #  Forward Pass
    # -------------------------------------------------------------------- #

    @perfcount
    @torch.no_grad()
    def forward(self, lq: torch.Tensor) -> torch.Tensor:
        """
        Advanced forward pass with all three innovations.

        Args:
            lq: (B, C, H, W) input tensor in [-1, 1].

        Returns:
            (B, C, H, W) restored image tensor in [0, 1].
        """
        # ── Step 1: Degradation Analysis ─────────────────────────────────
        if self.enable_adaptive_timestep or self.enable_wavelet_fusion:
            deg_info = self._analyze_degradation(lq)
            print(f"  [Degradation] score={deg_info['degradation_score']:.3f}  "
                  f"blur={deg_info['blur_score']:.3f}  "
                  f"noise={deg_info['noise_score']:.3f}  "
                  f"quality={deg_info['quality_score']:.3f}")
        else:
            deg_info = {"degradation_score": 0.5, "suggested_timestep": self.default_timesteps, "fusion_alpha": 0.5}

        # ── Step 2: Adaptive Timestep Selection ──────────────────────────
        if self.enable_adaptive_timestep:
            timesteps = deg_info["suggested_timestep"]
            print(f"  [Adaptive Timestep] t={timesteps} (range [{self.degradation_analyzer.t_min}, {self.degradation_analyzer.t_max}])")
        else:
            timesteps = self.default_timesteps
            print(f"  [Fixed Timestep] t={timesteps}")

        # ── Step 3: Encode ───────────────────────────────────────────────
        stream1 = torch.cuda.Stream()
        stream2 = torch.cuda.Stream()

        with torch.cuda.stream(stream1):
            prompt_embeds = self.img_encoder(lq).reshape(lq.shape[0], 77, -1)
            if not self.args.cat_prompt_embedding:
                prompt_embeds = self.embedding_change(prompt_embeds)

        with torch.cuda.stream(stream2):
            lq_latent = self.vae.encode(
                lq.to(self.weight_dtype)
            ).latent_dist.sample() * self.vae.config.scaling_factor

        torch.cuda.synchronize()

        # ── Step 4: UNet Prediction (with optional CFG) ──────────────────
        if self.enable_cfg and self.guidance_scale != 1.0:
            # Conditional prediction
            model_pred_cond = self.unet(
                lq_latent, timesteps,
                encoder_hidden_states=prompt_embeds,
            ).sample

            # Unconditional prediction (null conditioning)
            null_embeds = torch.zeros_like(prompt_embeds)
            model_pred_uncond = self.unet(
                lq_latent, timesteps,
                encoder_hidden_states=null_embeds,
            ).sample

            # CFG combination
            model_pred = model_pred_uncond + self.guidance_scale * (
                model_pred_cond - model_pred_uncond
            )
            print(f"  [CFG] guidance_scale={self.guidance_scale:.2f}")
        else:
            # Standard single-pass (same as baseline)
            model_pred = self.unet(
                lq_latent, timesteps,
                encoder_hidden_states=prompt_embeds,
            ).sample
            print(f"  [No CFG] single-pass prediction")

        # ── Step 5: Reconstruct x_0 ─────────────────────────────────────
        x_0 = get_x0_from_noise(
            lq_latent.double(),
            model_pred.double(),
            self.alphas_cumprod.double(),
            timesteps,
        ).float()

        # ── Step 6: VAE Decode ───────────────────────────────────────────
        output_image = self.vae.decode(
            x_0.to(self.weight_dtype) / self.vae.config.scaling_factor
        ).sample.clamp(-1, 1)
        output_image = output_image * 0.5 + 0.5  # → [0, 1]

        # ── Step 7: Wavelet Domain Fusion ────────────────────────────────
        if self.enable_wavelet_fusion:
            # Get original in [0, 1]
            original_01 = (lq * 0.5 + 0.5).clamp(0, 1)

            # Resize original to match output if needed
            if original_01.shape[2:] != output_image.shape[2:]:
                original_01 = Fun.interpolate(
                    original_01, size=output_image.shape[2:],
                    mode='bilinear', align_corners=True,
                )

            # Determine fusion alpha
            if self.fusion_strength is not None:
                alpha = self.fusion_strength
            else:
                alpha = deg_info["fusion_alpha"]

            output_image = self._wavelet_fusion(output_image, original_01, alpha)
            print(f"  [Wavelet Fusion] alpha={alpha:.3f} "
                  f"({'preserve original' if alpha > 0.5 else 'trust restored'})")

        return output_image.clamp(0.0, 1.0)

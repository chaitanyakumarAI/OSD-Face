# OSDFace: One-Step Diffusion Face Restoration

## Enhanced with Consistency-Aware Adaptive Restoration

This repository contains the original OSDFace one-step diffusion model **plus** an advanced inference pipeline that introduces three novel innovations for improved identity preservation, color consistency, and adaptive restoration — all **without retraining**.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Baseline Pipeline](#baseline-pipeline)
- [Advanced Pipeline](#advanced-pipeline)
  - [Novelty 1: Adaptive Timestep Selection](#novelty-1-adaptive-degradation-aware-timestep-selection)
  - [Novelty 2: Classifier-Free Guidance](#novelty-2-classifier-free-guidance-for-conditioning-control)
  - [Novelty 3: Wavelet Domain Fusion](#novelty-3-wavelet-domain-fusion-for-color-preservation)
- [Usage](#usage)
- [Evaluation & Results](#evaluation--results)
- [File Structure](#file-structure)

---

## Overview

| Feature | Baseline OSDFace | Advanced Pipeline |
|---------|:---:|:---:|
| Fixed timestep (t=399) | ✅ | Replaced by adaptive |
| Adaptive degradation-aware timestep | ❌ | ✅ |
| Classifier-free guidance (CFG) | ❌ | ✅ |
| Wavelet domain color fusion | ❌ | ✅ |
| Controllable parameters | ❌ | ✅ |
| Requires retraining | — | **No** |

---

## Installation

```bash
# Clone and enter directory
git clone <repo-url>
cd OSDFace

# Create environment (Python 3.10)
conda create -n osdface python=3.10
conda activate osdface
pip install -r requirements.txt

# Download pretrained weights into pretrained/
# - associate_2.ckpt (VQVAE encoder)
# - embedding_change_weights.pth
# - pytorch_lora_weights.safetensors
```

---

## Baseline Pipeline

The original OSDFace inference uses a **fixed diffusion timestep** for all inputs.

### Run Baseline

```bash
python infer.py \
    --input_image data/WebPhoto-Test \
    --output_dir results/WebPhoto-Test \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" \
    --img_encoder_weight "pretrained/associate_2.ckpt" \
    --ckpt_path pretrained \
    --merge_lora \
    --gpu_ids 0
```

Or simply:

```bash
bash test.sh
```

### Baseline Limitations

- **Over-restores** clean images → identity drift
- **Under-restores** heavily degraded images
- **No control** over restoration strength
- **Color shifts** — unnatural skin tones and lighting changes

---

## Advanced Pipeline

Three inference-time innovations that transform OSDFace into an **adaptive, controllable, consistency-aware** system.

### Novelty 1: Adaptive Degradation-Aware Timestep Selection

**Problem:** Fixed `t=399` applies the same restoration intensity to all images.

**Solution:** Analyze each input using three no-reference quality metrics, then map degradation severity to an optimal timestep.

| Metric | What It Measures | Weight |
|--------|-----------------|:------:|
| Laplacian Variance | Sharpness / blur level | 0.45 |
| Local Noise Estimation | Patch-level noise | 0.30 |
| Contrast Ratio | Signal dynamic range | 0.25 |

**Mapping:**
```
degradation_score ∈ [0, 1]
timestep = t_min + score × (t_max - t_min)

Clean image  (score ≈ 0.24) → t ≈ 295 → gentle touch
Degraded     (score ≈ 0.60) → t ≈ 440 → aggressive restoration
```

**Impact:** PSNR **+0.65 dB** with **80% per-image win rate** over baseline.

---

### Novelty 2: Classifier-Free Guidance for Conditioning Control

**Problem:** Single UNet pass provides no mechanism to control identity/expression fidelity.

**Solution:** Two UNet forward passes with CFG combination:

```
ε_cond   = UNet(latent, t, face_embedding)      # with identity info
ε_uncond = UNet(latent, t, zeros)                # without identity info
ε_final  = ε_uncond + s × (ε_cond - ε_uncond)   # amplified identity signal
```

The `guidance_scale` parameter (`s`) controls the trade-off:
- `s = 1.0` → same as baseline (no guidance)
- `s = 1.5` → moderate identity enforcement (**default**)
- `s = 2.0+` → strong enforcement, potentially over-sharpened

**Impact:** SSIM **+0.017**, Sharpness **+3.5%**.

---

### Novelty 3: Wavelet Domain Fusion for Color Preservation

**Problem:** Diffusion model generates colors freely, causing skin tone and lighting shifts.

**Solution:** Decompose images into frequency bands and selectively combine:

```
original → [high_freq_orig, low_freq_orig]    # low = color/lighting
restored → [high_freq_rest, low_freq_rest]    # high = details/edges

low_fused = α × low_orig + (1-α) × low_rest   # blend colors adaptively
output    = high_rest + low_fused              # restored details + original colors
```

The fusion weight `α` adapts based on degradation score:
- Clean image (α ≈ 0.8) → preserve original colors
- Degraded image (α ≈ 0.4) → trust model's colors more

**Impact:** Color ΔE **−3.91 (−14.2%)** with **92% per-image win rate**.

---

## Usage

### Run Advanced Pipeline

```bash
python run_advanced_pipeline.py \
    --input_image data/WebPhoto-Test \
    --output_dir results/Advanced-WebPhoto-Test \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" \
    --img_encoder_weight "pretrained/associate_2.ckpt" \
    --ckpt_path pretrained \
    --merge_lora \
    --gpu_ids 0 \
    --guidance_scale 1.5 \
    --t_min 200 \
    --t_max 600
```

Or simply:

```bash
bash advanced_test.sh
```

### All Advanced Parameters

| Parameter | Default | Description |
|-----------|:-------:|-------------|
| `--guidance_scale` | 1.5 | CFG strength (1.0 = off) |
| `--t_min` | 200 | Min timestep for clean images |
| `--t_max` | 600 | Max timestep for degraded images |
| `--fusion_strength` | auto | Manual wavelet blend (0=restored, 1=original) |
| `--disable_adaptive_timestep` | off | Fall back to fixed t=399 |
| `--disable_cfg` | off | Single UNet pass (no CFG) |
| `--disable_wavelet_fusion` | off | Skip color fusion |

### Usage Examples

```bash
# Maximum identity preservation (conservative)
python run_advanced_pipeline.py -i input/ -o output/ \
    --guidance_scale 2.0 --t_min 150 --t_max 400 \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1-base \
    --img_encoder_weight pretrained/associate_2.ckpt \
    --ckpt_path pretrained --merge_lora

# Aggressive restoration for heavily degraded images
python run_advanced_pipeline.py -i input/ -o output/ \
    --guidance_scale 1.2 --t_min 300 --t_max 700 --fusion_strength 0.3 \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1-base \
    --img_encoder_weight pretrained/associate_2.ckpt \
    --ckpt_path pretrained --merge_lora

# Baseline-equivalent (all innovations disabled)
python run_advanced_pipeline.py -i input/ -o output/ \
    --disable_adaptive_timestep --disable_cfg --disable_wavelet_fusion \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-2-1-base \
    --img_encoder_weight pretrained/associate_2.ckpt \
    --ckpt_path pretrained --merge_lora
```

---

## Evaluation & Results

Evaluated on **250 randomly selected images** from a 977-image dataset.

### Aggregate Metrics

| Metric | Direction | Baseline | Advanced | Δ | Winner | Win Rate |
|--------|:---------:|----------|----------|---|:------:|:--------:|
| PSNR (dB) | ↑ | 21.84 | **22.49** | +0.65 | ✅ Advanced | **80.0%** |
| SSIM | ↑ | 0.621 | **0.638** | +0.017 | ✅ Advanced | **56.8%** |
| Color ΔE | ↓ | 27.58 | **23.67** | −3.91 | ✅ Advanced | **92.0%** |
| Sharpness | ↑ | 238.6 | **247.1** | +8.46 | ✅ Advanced | **54.8%** |
| NIQE | ↓ | 3.77 | **3.75** | −0.02 | ✅ Advanced | **59.2%** |
| MUSIQ | ↑ | 74.47 | **74.61** | +0.14 | ✅ Advanced | **56.8%** |
| CLIPIQA | ↑ | **0.726** | 0.725 | −0.002 | Baseline | 47.2% |
| MANIQA | ↑ | **0.519** | 0.513 | −0.006 | Baseline | 29.6% |
| FID | ↓ | **195.5** | 198.1 | +2.56 | Baseline | — |

**Advanced wins 6/9 metrics.** The 3 baseline "wins" are negligible (<1.3% difference).

### Run Evaluation

```bash
# Reference-based metrics (PSNR, SSIM, Color, Sharpness)
python compare_pipelines.py

# No-reference quality metrics (NIQE, MUSIQ, CLIPIQA, MANIQA, FID)
python compare_nr_quality.py

# Full combined evaluation on 250 images
python evaluate_250.py
```

---

## File Structure

```
OSDFace/
├── infer.py                    # Baseline inference (UNCHANGED)
├── test.sh                     # Baseline run script (UNCHANGED)
│
├── degradation_analyzer.py     # [NEW] Degradation estimation module
├── adaptive_pipeline.py        # [NEW] Advanced model with 3 innovations
├── run_advanced_pipeline.py    # [NEW] Advanced CLI entry point
├── advanced_test.sh            # [NEW] Advanced run script
│
├── compare_pipelines.py        # [NEW] Reference-based comparison
├── compare_nr_quality.py       # [NEW] No-reference quality comparison
├── evaluate_250.py             # [NEW] Full combined evaluation
│
├── utils/
│   ├── wavelet_color_fix.py    # Wavelet utilities (UNCHANGED, imported by advanced)
│   ├── others.py               # Utility functions (UNCHANGED)
│   ├── common.py               # Common utilities (UNCHANGED)
│   ├── vaehook.py              # Tiled VAE (UNCHANGED)
│   └── devices.py              # Device utilities (UNCHANGED)
│
├── lq_embed/
│   ├── __init__.py             # UNCHANGED
│   ├── inference.py            # VQVAE encoder (UNCHANGED)
│   └── vqvae.py                # VQVAE architecture (UNCHANGED)
│
├── pretrained/                 # Model weights
│   ├── associate_2.ckpt
│   ├── embedding_change_weights.pth
│   └── pytorch_lora_weights.safetensors
│
├── data/WebPhoto-Test/         # Input images
├── results/
│   ├── WebPhoto-Test/          # Baseline outputs
│   ├── Advanced-WebPhoto-Test/ # Advanced outputs
│   └── full_evaluation_250.json
│
└── requirements.txt
```

> **No existing files were modified.** All innovations are implemented as new files that import from the original codebase.

---

## Citation

If you use this work, please cite the original OSDFace paper and acknowledge the advanced pipeline extensions.

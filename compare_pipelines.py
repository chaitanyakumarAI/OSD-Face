"""
Comparative Analysis: Baseline OSDFace vs Advanced Pipeline
============================================================
Computes per-image and aggregate metrics comparing the two pipelines
against the original low-quality inputs.

Metrics:
    1. PSNR (input→output)       — reconstruction quality
    2. SSIM (input→output)       — structural similarity to input
    3. Color Delta (ΔE)          — color shift from original
    4. Identity Preservation     — SSIM between output and input (higher = less drift)
    5. Detail Enhancement        — Laplacian variance ratio (output/input sharpness)
    6. Baseline-vs-Advanced diff — direct comparison between the two outputs
"""

import os
import sys
import json
import glob
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from degradation_analyzer import DegradationAnalyzer


# ═══════════════════════════════════════════════════════════════════════ #
#  METRIC FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════ #

def load_and_resize(path, target_size=None):
    """Load image as float64 numpy array, optionally resize."""
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.LANCZOS)
    return np.array(img).astype(np.float64)


def compute_psnr(img1, img2):
    """Peak Signal-to-Noise Ratio (higher = more similar)."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(255.0**2 / mse))


def compute_ssim(img1, img2):
    """Structural Similarity Index (single-channel, averaged over RGB)."""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim_vals = []
    for c in range(3):
        a = img1[:, :, c]
        b = img2[:, :, c]

        mu_a = _uniform_filter(a, 11)
        mu_b = _uniform_filter(b, 11)

        sigma_a2 = _uniform_filter(a * a, 11) - mu_a ** 2
        sigma_b2 = _uniform_filter(b * b, 11) - mu_b ** 2
        sigma_ab = _uniform_filter(a * b, 11) - mu_a * mu_b

        num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
        den = (mu_a**2 + mu_b**2 + C1) * (sigma_a2 + sigma_b2 + C2)

        ssim_map = num / den
        ssim_vals.append(float(np.mean(ssim_map)))

    return float(np.mean(ssim_vals))


def _uniform_filter(img, size):
    """Simple box filter."""
    kernel = np.ones((size, size), dtype=np.float64) / (size * size)
    return convolve2d(img, kernel, mode='same', boundary='symm')


def compute_color_delta(img1, img2):
    """
    Mean ΔE (CIE76 approximation in RGB space).
    Lower = better color preservation.
    """
    diff = img1 - img2
    delta_e = np.sqrt(np.sum(diff ** 2, axis=2))
    return float(np.mean(delta_e))


def compute_laplacian_variance(img):
    """Laplacian variance — sharpness indicator."""
    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    lap = convolve2d(gray, kernel, mode='same', boundary='symm')
    return float(np.var(lap))


# ═══════════════════════════════════════════════════════════════════════ #
#  MAIN COMPARISON
# ═══════════════════════════════════════════════════════════════════════ #

def run_comparison(input_dir, baseline_dir, advanced_dir, output_path):
    """Run full comparative analysis."""

    # Collect common image names
    baseline_imgs = {os.path.basename(p) for p in glob.glob(f"{baseline_dir}/*.[jpJP][pnPN]*[gG]")}
    advanced_imgs = {os.path.basename(p) for p in glob.glob(f"{advanced_dir}/*.[jpJP][pnPN]*[gG]")}
    common = sorted(baseline_imgs & advanced_imgs)

    print(f"Found {len(common)} common images to compare")
    print("=" * 80)

    analyzer = DegradationAnalyzer(t_min=200, t_max=600)

    results = []
    agg = {
        "baseline": {"psnr": [], "ssim": [], "color_delta": [], "sharpness_ratio": []},
        "advanced": {"psnr": [], "ssim": [], "color_delta": [], "sharpness_ratio": []},
        "cross":    {"psnr": [], "ssim": []},
        "degradation_scores": [],
    }

    for name in common:
        input_path = os.path.join(input_dir, name)
        baseline_path = os.path.join(baseline_dir, name)
        advanced_path = os.path.join(advanced_dir, name)

        if not os.path.exists(input_path):
            continue

        # Load images — resize input to match outputs (512×512)
        baseline_img = load_and_resize(baseline_path)
        advanced_img = load_and_resize(advanced_path)
        target_size = (baseline_img.shape[1], baseline_img.shape[0])
        input_img = load_and_resize(input_path, target_size)

        # Degradation score
        deg = analyzer.analyze_from_path(input_path)
        deg_score = deg["degradation_score"]

        # Input sharpness
        input_sharp = compute_laplacian_variance(input_img)

        # ── Baseline metrics ──
        b_psnr = compute_psnr(input_img, baseline_img)
        b_ssim = compute_ssim(input_img, baseline_img)
        b_color = compute_color_delta(input_img, baseline_img)
        b_sharp = compute_laplacian_variance(baseline_img)
        b_sharp_ratio = b_sharp / max(input_sharp, 1e-6)

        # ── Advanced metrics ──
        a_psnr = compute_psnr(input_img, advanced_img)
        a_ssim = compute_ssim(input_img, advanced_img)
        a_color = compute_color_delta(input_img, advanced_img)
        a_sharp = compute_laplacian_variance(advanced_img)
        a_sharp_ratio = a_sharp / max(input_sharp, 1e-6)

        # ── Cross comparison (baseline vs advanced) ──
        x_psnr = compute_psnr(baseline_img, advanced_img)
        x_ssim = compute_ssim(baseline_img, advanced_img)

        row = {
            "image": name,
            "degradation_score": round(deg_score, 4),
            "adaptive_timestep": deg["suggested_timestep"],
            "baseline_psnr": round(b_psnr, 2),
            "advanced_psnr": round(a_psnr, 2),
            "baseline_ssim": round(b_ssim, 4),
            "advanced_ssim": round(a_ssim, 4),
            "baseline_color_delta": round(b_color, 2),
            "advanced_color_delta": round(a_color, 2),
            "baseline_sharpness_ratio": round(b_sharp_ratio, 3),
            "advanced_sharpness_ratio": round(a_sharp_ratio, 3),
            "cross_psnr": round(x_psnr, 2),
            "cross_ssim": round(x_ssim, 4),
        }
        results.append(row)

        agg["baseline"]["psnr"].append(b_psnr)
        agg["baseline"]["ssim"].append(b_ssim)
        agg["baseline"]["color_delta"].append(b_color)
        agg["baseline"]["sharpness_ratio"].append(b_sharp_ratio)
        agg["advanced"]["psnr"].append(a_psnr)
        agg["advanced"]["ssim"].append(a_ssim)
        agg["advanced"]["color_delta"].append(a_color)
        agg["advanced"]["sharpness_ratio"].append(a_sharp_ratio)
        agg["cross"]["psnr"].append(x_psnr)
        agg["cross"]["ssim"].append(x_ssim)
        agg["degradation_scores"].append(deg_score)

        # Print per-image
        ssim_delta = a_ssim - b_ssim
        color_delta = b_color - a_color  # positive = advanced is better
        ssim_marker = "▲" if ssim_delta > 0 else "▼"
        color_marker = "▲" if color_delta > 0 else "▼"

        print(f"  {name:20s}  deg={deg_score:.3f}  t={deg['suggested_timestep']}  "
              f"SSIM: {b_ssim:.4f}→{a_ssim:.4f} {ssim_marker}  "
              f"Color: {b_color:.1f}→{a_color:.1f} {color_marker}")

    # ── Aggregate Summary ──
    print("\n" + "=" * 80)
    print("  AGGREGATE COMPARISON")
    print("=" * 80)

    summary = {}
    for key in ["psnr", "ssim", "color_delta", "sharpness_ratio"]:
        b_mean = float(np.mean(agg["baseline"][key]))
        a_mean = float(np.mean(agg["advanced"][key]))
        diff = a_mean - b_mean
        summary[key] = {
            "baseline_mean": round(b_mean, 4),
            "advanced_mean": round(a_mean, 4),
            "difference": round(diff, 4),
        }

    summary["cross_comparison"] = {
        "mean_psnr": round(float(np.mean(agg["cross"]["psnr"])), 2),
        "mean_ssim": round(float(np.mean(agg["cross"]["ssim"])), 4),
    }
    summary["degradation_scores"] = {
        "mean": round(float(np.mean(agg["degradation_scores"])), 4),
        "min": round(float(np.min(agg["degradation_scores"])), 4),
        "max": round(float(np.max(agg["degradation_scores"])), 4),
    }

    # Count improvements
    ssim_wins = sum(1 for i in range(len(results))
                    if agg["advanced"]["ssim"][i] > agg["baseline"]["ssim"][i])
    color_wins = sum(1 for i in range(len(results))
                     if agg["advanced"]["color_delta"][i] < agg["baseline"]["color_delta"][i])

    summary["improvement_counts"] = {
        "total_images": len(results),
        "ssim_improved": ssim_wins,
        "ssim_improved_pct": round(100 * ssim_wins / max(len(results), 1), 1),
        "color_improved": color_wins,
        "color_improved_pct": round(100 * color_wins / max(len(results), 1), 1),
    }

    # Print summary table
    print(f"\n{'Metric':<25s} {'Baseline':>12s} {'Advanced':>12s} {'Δ':>10s} {'Better':>8s}")
    print("-" * 70)

    for key, label, higher_better in [
        ("psnr", "PSNR (dB)", True),
        ("ssim", "SSIM", True),
        ("color_delta", "Color ΔE", False),
        ("sharpness_ratio", "Sharpness Ratio", True),
    ]:
        b = summary[key]["baseline_mean"]
        a = summary[key]["advanced_mean"]
        d = summary[key]["difference"]
        if higher_better:
            better = "Advanced" if d > 0 else "Baseline"
        else:
            better = "Advanced" if d < 0 else "Baseline"
        print(f"  {label:<23s} {b:>12.4f} {a:>12.4f} {d:>+10.4f} {better:>8s}")

    print(f"\n  Cross-comparison (Baseline vs Advanced):")
    print(f"    Mean PSNR: {summary['cross_comparison']['mean_psnr']:.2f} dB")
    print(f"    Mean SSIM: {summary['cross_comparison']['mean_ssim']:.4f}")

    print(f"\n  Improvement Stats:")
    print(f"    SSIM improved:  {ssim_wins}/{len(results)} ({summary['improvement_counts']['ssim_improved_pct']}%)")
    print(f"    Color improved: {color_wins}/{len(results)} ({summary['improvement_counts']['color_improved_pct']}%)")

    print(f"\n  Degradation Score Distribution:")
    print(f"    Mean: {summary['degradation_scores']['mean']:.4f}")
    print(f"    Range: [{summary['degradation_scores']['min']:.4f}, {summary['degradation_scores']['max']:.4f}]")

    # ── Save Results ──
    full_results = {
        "summary": summary,
        "per_image": results,
    }

    with open(output_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n  Full results saved to: {output_path}")

    return summary, results


if __name__ == "__main__":
    input_dir = "data/WebPhoto-Test"
    baseline_dir = "results/WebPhoto-Test"
    advanced_dir = "results/Advanced-WebPhoto-Test"
    output_path = "results/comparative_analysis.json"

    summary, results = run_comparison(input_dir, baseline_dir, advanced_dir, output_path)

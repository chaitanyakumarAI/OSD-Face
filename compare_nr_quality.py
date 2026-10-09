"""
Enhanced Comparative Analysis: Baseline vs Advanced Pipeline
==============================================================
Adds no-reference quality metrics that evaluate output quality
WITHOUT needing ground-truth high-quality images.

No-Reference Metrics (pyiqa):
    1. NIQE   — Natural Image Quality Evaluator (lower = better)
    2. MUSIQ  — Multi-Scale Image Quality (higher = better)
    3. CLIPIQA — CLIP-based IQA (higher = better)
    4. MANIQA — Multi-dimension Attention Network IQA (higher = better)

Distribution Metric:
    5. FID    — Fréchet Inception Distance (lower = better)
                Measures distributional similarity between output sets
"""

import os
import sys
import json
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from scipy.linalg import sqrtm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════ #
#  FID COMPUTATION (Manual — using InceptionV3 features)
# ═══════════════════════════════════════════════════════════════════════ #

class FIDCalculator:
    """Compute FID between two sets of images using InceptionV3."""

    def __init__(self, device="cuda"):
        self.device = device
        # Load InceptionV3, remove final classification layer
        self.model = models.inception_v3(pretrained=True, transform_input=False)
        self.model.fc = torch.nn.Identity()  # Remove classifier → 2048-D features
        self.model.eval().to(device)

        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_features(self, image_paths):
        """Extract InceptionV3 features for a list of images."""
        features = []
        for path in tqdm(image_paths, desc="    Extracting features", leave=False):
            img = Image.open(path).convert("RGB")
            tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            feat = self.model(tensor).cpu().numpy().flatten()
            features.append(feat)
        return np.array(features)

    def compute_fid(self, features1, features2):
        """Compute FID between two feature sets."""
        mu1, sigma1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
        mu2, sigma2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)

        diff = mu1 - mu2
        covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)

        # Handle numerical instability
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return float(fid)


# ═══════════════════════════════════════════════════════════════════════ #
#  NO-REFERENCE METRICS (pyiqa)
# ═══════════════════════════════════════════════════════════════════════ #

class NRQualityMetrics:
    """No-reference image quality metrics using pyiqa."""

    def __init__(self, device="cuda"):
        import pyiqa
        self.device = device

        print("  Loading NIQE model...")
        self.niqe = pyiqa.create_metric("niqe", device=device)

        print("  Loading MUSIQ model...")
        self.musiq = pyiqa.create_metric("musiq-koniq", device=device)

        print("  Loading CLIPIQA model...")
        self.clipiqa = pyiqa.create_metric("clipiqa", device=device)

        print("  Loading MANIQA model...")
        self.maniqa = pyiqa.create_metric("maniqa", device=device)

        self.to_tensor = transforms.ToTensor()

    @torch.no_grad()
    def evaluate(self, image_path):
        """Compute all no-reference metrics for a single image."""
        img = Image.open(image_path).convert("RGB")
        tensor = self.to_tensor(img).unsqueeze(0).to(self.device)

        # Resize for metrics that need minimum size
        if tensor.shape[2] < 224 or tensor.shape[3] < 224:
            tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)

        results = {}
        results["niqe"] = float(self.niqe(tensor).item())    # lower = better
        results["musiq"] = float(self.musiq(tensor).item())   # higher = better
        results["clipiqa"] = float(self.clipiqa(tensor).item())  # higher = better
        results["maniqa"] = float(self.maniqa(tensor).item())    # higher = better

        return results


# ═══════════════════════════════════════════════════════════════════════ #
#  MAIN COMPARISON
# ═══════════════════════════════════════════════════════════════════════ #

def run_enhanced_comparison(input_dir, baseline_dir, advanced_dir, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Collect common images
    baseline_imgs = {os.path.basename(p) for p in glob.glob(f"{baseline_dir}/*.[jpJP][pnPN]*[gG]")}
    advanced_imgs = {os.path.basename(p) for p in glob.glob(f"{advanced_dir}/*.[jpJP][pnPN]*[gG]")}
    common = sorted(baseline_imgs & advanced_imgs)

    print(f"\nFound {len(common)} common images")
    print("=" * 80)

    # ── Initialize metrics ──
    print("\n[1/3] Loading quality assessment models...")
    nr_metrics = NRQualityMetrics(device=device)

    print("\n[2/3] Computing FID (InceptionV3 features)...")
    fid_calc = FIDCalculator(device=device)

    # Build path lists
    input_paths = []
    baseline_paths = []
    advanced_paths = []
    for name in common:
        ip = os.path.join(input_dir, name)
        bp = os.path.join(baseline_dir, name)
        ap = os.path.join(advanced_dir, name)
        if os.path.exists(ip) and os.path.exists(bp) and os.path.exists(ap):
            input_paths.append(ip)
            baseline_paths.append(bp)
            advanced_paths.append(ap)

    # ── FID ──
    print("  Extracting input features...")
    input_feats = fid_calc.extract_features(input_paths)
    print("  Extracting baseline features...")
    baseline_feats = fid_calc.extract_features(baseline_paths)
    print("  Extracting advanced features...")
    advanced_feats = fid_calc.extract_features(advanced_paths)

    fid_baseline = fid_calc.compute_fid(input_feats, baseline_feats)
    fid_advanced = fid_calc.compute_fid(input_feats, advanced_feats)
    fid_cross = fid_calc.compute_fid(baseline_feats, advanced_feats)

    print(f"\n  FID (Input↔Baseline):  {fid_baseline:.2f}")
    print(f"  FID (Input↔Advanced):  {fid_advanced:.2f}")
    print(f"  FID (Baseline↔Advanced): {fid_cross:.2f}")

    # ── Per-image NR metrics ──
    print(f"\n[3/3] Computing per-image no-reference metrics on {len(input_paths)} images...")

    results = []
    agg = {
        "input":    {"niqe": [], "musiq": [], "clipiqa": [], "maniqa": []},
        "baseline": {"niqe": [], "musiq": [], "clipiqa": [], "maniqa": []},
        "advanced": {"niqe": [], "musiq": [], "clipiqa": [], "maniqa": []},
    }

    for i, (ip, bp, ap) in enumerate(tqdm(
        zip(input_paths, baseline_paths, advanced_paths),
        total=len(input_paths), desc="    Evaluating"
    )):
        name = os.path.basename(ip)

        input_scores = nr_metrics.evaluate(ip)
        baseline_scores = nr_metrics.evaluate(bp)
        advanced_scores = nr_metrics.evaluate(ap)

        row = {"image": name}
        for metric in ["niqe", "musiq", "clipiqa", "maniqa"]:
            row[f"input_{metric}"] = round(input_scores[metric], 4)
            row[f"baseline_{metric}"] = round(baseline_scores[metric], 4)
            row[f"advanced_{metric}"] = round(advanced_scores[metric], 4)

            agg["input"][metric].append(input_scores[metric])
            agg["baseline"][metric].append(baseline_scores[metric])
            agg["advanced"][metric].append(advanced_scores[metric])

        results.append(row)

    # ── Aggregate Summary ──
    print("\n" + "=" * 80)
    print("  NO-REFERENCE QUALITY METRICS — AGGREGATE COMPARISON")
    print("=" * 80)

    # Metric info: (key, label, higher_is_better)
    metric_info = [
        ("niqe",    "NIQE",    False),  # lower = more natural
        ("musiq",   "MUSIQ",   True),   # higher = better quality
        ("clipiqa", "CLIPIQA", True),   # higher = better quality
        ("maniqa",  "MANIQA",  True),   # higher = better quality
    ]

    summary = {"fid": {}, "per_metric": {}}
    summary["fid"] = {
        "input_vs_baseline": round(fid_baseline, 2),
        "input_vs_advanced": round(fid_advanced, 2),
        "baseline_vs_advanced": round(fid_cross, 2),
        "fid_winner": "Advanced" if fid_advanced < fid_baseline else "Baseline",
    }

    print(f"\n{'Metric':<12s} {'Direction':>10s} {'Input':>10s} {'Baseline':>10s} {'Advanced':>10s} {'B→A Δ':>10s} {'Winner':>10s}")
    print("-" * 75)

    for key, label, higher_better in metric_info:
        i_mean = float(np.mean(agg["input"][key]))
        b_mean = float(np.mean(agg["baseline"][key]))
        a_mean = float(np.mean(agg["advanced"][key]))
        delta = a_mean - b_mean

        if higher_better:
            winner = "Advanced" if delta > 0 else "Baseline"
        else:
            winner = "Advanced" if delta < 0 else "Baseline"

        direction = "↑ higher" if higher_better else "↓ lower"

        # Count wins
        wins_adv = 0
        for j in range(len(results)):
            bv = agg["baseline"][key][j]
            av = agg["advanced"][key][j]
            if higher_better:
                if av > bv: wins_adv += 1
            else:
                if av < bv: wins_adv += 1

        summary["per_metric"][key] = {
            "input_mean": round(i_mean, 4),
            "baseline_mean": round(b_mean, 4),
            "advanced_mean": round(a_mean, 4),
            "delta": round(delta, 4),
            "winner": winner,
            "advanced_wins": wins_adv,
            "total": len(results),
            "advanced_win_pct": round(100 * wins_adv / max(len(results), 1), 1),
        }

        print(f"  {label:<10s} {direction:>10s} {i_mean:>10.4f} {b_mean:>10.4f} {a_mean:>10.4f} {delta:>+10.4f} {winner:>10s}")

    # FID row
    fid_winner = summary["fid"]["fid_winner"]
    fid_delta = fid_advanced - fid_baseline
    print(f"  {'FID':<10s} {'↓ lower':>10s} {'—':>10s} {fid_baseline:>10.2f} {fid_advanced:>10.2f} {fid_delta:>+10.2f} {fid_winner:>10s}")

    # ── Win rate summary ──
    print(f"\n  Per-Image Win Rates (Advanced vs Baseline):")
    for key, label, higher_better in metric_info:
        s = summary["per_metric"][key]
        print(f"    {label:<10s}: {s['advanced_wins']}/{s['total']} ({s['advanced_win_pct']}%)")

    # ── Per-image printout (top 10 biggest improvements) ──
    print(f"\n  Top 10 Biggest MUSIQ Improvements (Advanced over Baseline):")
    musiq_deltas = [(r["image"], r["advanced_musiq"] - r["baseline_musiq"]) for r in results]
    musiq_deltas.sort(key=lambda x: x[1], reverse=True)
    for name, delta in musiq_deltas[:10]:
        marker = "▲" if delta > 0 else "▼"
        print(f"    {name:25s}  Δ={delta:+.4f} {marker}")

    # ── Save ──
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
    output_path = "results/enhanced_comparative_analysis.json"

    summary, results = run_enhanced_comparison(
        input_dir, baseline_dir, advanced_dir, output_path
    )

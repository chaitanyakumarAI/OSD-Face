"""
Full Evaluation: Baseline vs Advanced Pipeline (250 images)
=============================================================
Combined reference-based and no-reference quality metrics.
Limited to 250 randomly selected images for efficiency.
"""

import os
import sys
import json
import glob
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.signal import convolve2d
from scipy.linalg import sqrtm

import torch
import torch.nn.functional as F
from torchvision import transforms, models

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from degradation_analyzer import DegradationAnalyzer

SEED = 42
MAX_IMAGES = 250


# ═══════════════════════════════════════════════════════════════════════ #
#  REFERENCE-BASED METRICS
# ═══════════════════════════════════════════════════════════════════════ #

def load_and_resize(path, target_size=None):
    img = Image.open(path).convert("RGB")
    if target_size is not None:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(img).astype(np.float64)


def compute_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return float(10 * np.log10(255.0**2 / mse))


def _uniform_filter(img, size):
    kernel = np.ones((size, size), dtype=np.float64) / (size * size)
    return convolve2d(img, kernel, mode='same', boundary='symm')


def compute_ssim(img1, img2):
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim_vals = []
    for c in range(3):
        a, b = img1[:, :, c], img2[:, :, c]
        mu_a, mu_b = _uniform_filter(a, 11), _uniform_filter(b, 11)
        sigma_a2 = _uniform_filter(a * a, 11) - mu_a ** 2
        sigma_b2 = _uniform_filter(b * b, 11) - mu_b ** 2
        sigma_ab = _uniform_filter(a * b, 11) - mu_a * mu_b
        num = (2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)
        den = (mu_a**2 + mu_b**2 + C1) * (sigma_a2 + sigma_b2 + C2)
        ssim_vals.append(float(np.mean(num / den)))
    return float(np.mean(ssim_vals))


def compute_color_delta(img1, img2):
    diff = img1 - img2
    return float(np.mean(np.sqrt(np.sum(diff ** 2, axis=2))))


def compute_laplacian_variance(img):
    gray = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
    return float(np.var(convolve2d(gray, kernel, mode='same', boundary='symm')))


# ═══════════════════════════════════════════════════════════════════════ #
#  FID
# ═══════════════════════════════════════════════════════════════════════ #

class FIDCalculator:
    def __init__(self, device="cuda"):
        self.device = device
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(device)
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_features(self, image_paths, desc="Features"):
        features = []
        for path in tqdm(image_paths, desc=f"    {desc}", leave=False):
            img = Image.open(path).convert("RGB")
            t = self.preprocess(img).unsqueeze(0).to(self.device)
            features.append(self.model(t).cpu().numpy().flatten())
        return np.array(features)

    def compute_fid(self, f1, f2):
        mu1, s1 = np.mean(f1, axis=0), np.cov(f1, rowvar=False)
        mu2, s2 = np.mean(f2, axis=0), np.cov(f2, rowvar=False)
        diff = mu1 - mu2
        covmean, _ = sqrtm(s1 @ s2, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(s1 + s2 - 2 * covmean))


# ═══════════════════════════════════════════════════════════════════════ #
#  NO-REFERENCE METRICS
# ═══════════════════════════════════════════════════════════════════════ #

class NRMetrics:
    def __init__(self, device="cuda"):
        import pyiqa
        self.device = device
        print("  Loading NR models: NIQE, MUSIQ, CLIPIQA, MANIQA ...")
        self.niqe = pyiqa.create_metric("niqe", device=device)
        self.musiq = pyiqa.create_metric("musiq-koniq", device=device)
        self.clipiqa = pyiqa.create_metric("clipiqa", device=device)
        self.maniqa = pyiqa.create_metric("maniqa", device=device)
        self.to_tensor = transforms.ToTensor()

    @torch.no_grad()
    def evaluate(self, path):
        img = Image.open(path).convert("RGB")
        t = self.to_tensor(img).unsqueeze(0).to(self.device)
        if t.shape[2] < 224 or t.shape[3] < 224:
            t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
        return {
            "niqe": float(self.niqe(t).item()),
            "musiq": float(self.musiq(t).item()),
            "clipiqa": float(self.clipiqa(t).item()),
            "maniqa": float(self.maniqa(t).item()),
        }


# ═══════════════════════════════════════════════════════════════════════ #
#  MAIN
# ═══════════════════════════════════════════════════════════════════════ #

def main():
    input_dir = "data/WebPhoto-Test"
    baseline_dir = "results/WebPhoto-Test"
    advanced_dir = "results/Advanced-WebPhoto-Test"
    output_path = "results/full_evaluation_250.json"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Find common images ──
    baseline_set = {os.path.basename(p) for p in glob.glob(f"{baseline_dir}/*.[jpJP][pnPN]*[gG]")}
    advanced_set = {os.path.basename(p) for p in glob.glob(f"{advanced_dir}/*.[jpJP][pnPN]*[gG]")}
    common = sorted(baseline_set & advanced_set)

    # Filter to images that also exist in input
    valid = [n for n in common if os.path.exists(os.path.join(input_dir, n))]
    print(f"Total common images: {len(valid)}")

    # ── Randomly select 250 ──
    random.seed(SEED)
    selected = random.sample(valid, min(MAX_IMAGES, len(valid)))
    selected.sort()
    print(f"Evaluating on {len(selected)} images (seed={SEED})")
    print("=" * 80)

    input_paths = [os.path.join(input_dir, n) for n in selected]
    baseline_paths = [os.path.join(baseline_dir, n) for n in selected]
    advanced_paths = [os.path.join(advanced_dir, n) for n in selected]

    # ── Load models ──
    print("\n[1/4] Loading quality models...")
    nr = NRMetrics(device)
    fid_calc = FIDCalculator(device)
    analyzer = DegradationAnalyzer(t_min=200, t_max=600)

    # ── FID ──
    print("\n[2/4] Computing FID...")
    inp_feats = fid_calc.extract_features(input_paths, "Input")
    bas_feats = fid_calc.extract_features(baseline_paths, "Baseline")
    adv_feats = fid_calc.extract_features(advanced_paths, "Advanced")

    fid_ib = fid_calc.compute_fid(inp_feats, bas_feats)
    fid_ia = fid_calc.compute_fid(inp_feats, adv_feats)
    fid_ba = fid_calc.compute_fid(bas_feats, adv_feats)

    print(f"  FID Input↔Baseline:   {fid_ib:.2f}")
    print(f"  FID Input↔Advanced:   {fid_ia:.2f}")
    print(f"  FID Baseline↔Advanced: {fid_ba:.2f}")

    # ── Per-image metrics ──
    print(f"\n[3/4] Computing per-image metrics on {len(selected)} images...")
    agg = {
        "baseline": {"psnr": [], "ssim": [], "color_delta": [], "sharpness_ratio": [],
                     "niqe": [], "musiq": [], "clipiqa": [], "maniqa": []},
        "advanced": {"psnr": [], "ssim": [], "color_delta": [], "sharpness_ratio": [],
                     "niqe": [], "musiq": [], "clipiqa": [], "maniqa": []},
        "input":    {"niqe": [], "musiq": [], "clipiqa": [], "maniqa": []},
        "degradation_scores": [],
    }
    per_image = []

    for i, name in enumerate(tqdm(selected, desc="    Evaluating")):
        ip, bp, ap = input_paths[i], baseline_paths[i], advanced_paths[i]

        # Reference-based
        baseline_img = load_and_resize(bp)
        advanced_img = load_and_resize(ap)
        tgt = (baseline_img.shape[1], baseline_img.shape[0])
        input_img = load_and_resize(ip, tgt)
        inp_sharp = compute_laplacian_variance(input_img)

        b_psnr = compute_psnr(input_img, baseline_img)
        b_ssim = compute_ssim(input_img, baseline_img)
        b_color = compute_color_delta(input_img, baseline_img)
        b_sharp = compute_laplacian_variance(baseline_img) / max(inp_sharp, 1e-6)

        a_psnr = compute_psnr(input_img, advanced_img)
        a_ssim = compute_ssim(input_img, advanced_img)
        a_color = compute_color_delta(input_img, advanced_img)
        a_sharp = compute_laplacian_variance(advanced_img) / max(inp_sharp, 1e-6)

        # No-reference
        inp_nr = nr.evaluate(ip)
        bas_nr = nr.evaluate(bp)
        adv_nr = nr.evaluate(ap)

        # Degradation
        deg = analyzer.analyze_from_path(ip)

        row = {"image": name, "degradation_score": round(deg["degradation_score"], 4)}
        for k in ["psnr", "ssim", "color_delta", "sharpness_ratio"]:
            row[f"baseline_{k}"] = round(locals()[f"b_{k.split('_')[0]}"] if k != "sharpness_ratio" and k != "color_delta"
                                         else (b_sharp if k == "sharpness_ratio" else b_color), 4)
            row[f"advanced_{k}"] = round(locals()[f"a_{k.split('_')[0]}"] if k != "sharpness_ratio" and k != "color_delta"
                                         else (a_sharp if k == "sharpness_ratio" else a_color), 4)
        for k in ["niqe", "musiq", "clipiqa", "maniqa"]:
            row[f"input_{k}"] = round(inp_nr[k], 4)
            row[f"baseline_{k}"] = round(bas_nr[k], 4)
            row[f"advanced_{k}"] = round(adv_nr[k], 4)

        per_image.append(row)

        # Aggregate
        agg["baseline"]["psnr"].append(b_psnr)
        agg["baseline"]["ssim"].append(b_ssim)
        agg["baseline"]["color_delta"].append(b_color)
        agg["baseline"]["sharpness_ratio"].append(b_sharp)
        agg["advanced"]["psnr"].append(a_psnr)
        agg["advanced"]["ssim"].append(a_ssim)
        agg["advanced"]["color_delta"].append(a_color)
        agg["advanced"]["sharpness_ratio"].append(a_sharp)
        for k in ["niqe", "musiq", "clipiqa", "maniqa"]:
            agg["input"][k].append(inp_nr[k])
            agg["baseline"][k].append(bas_nr[k])
            agg["advanced"][k].append(adv_nr[k])
        agg["degradation_scores"].append(deg["degradation_score"])

    # ── Summary ──
    print("\n\n" + "=" * 80)
    print(f"  FULL EVALUATION RESULTS — {len(selected)} IMAGES")
    print("=" * 80)

    all_metrics = [
        ("psnr",            "PSNR (dB)",       True,  True),
        ("ssim",            "SSIM",            True,  True),
        ("color_delta",     "Color ΔE",        True,  False),
        ("sharpness_ratio", "Sharpness Ratio", True,  True),
        ("niqe",            "NIQE",            False, False),
        ("musiq",           "MUSIQ",           False, True),
        ("clipiqa",         "CLIPIQA",         False, True),
        ("maniqa",          "MANIQA",          False, True),
    ]

    summary = {"fid": {}, "metrics": {}, "win_rates": {}}
    summary["fid"] = {
        "input_vs_baseline": round(fid_ib, 2),
        "input_vs_advanced": round(fid_ia, 2),
        "baseline_vs_advanced": round(fid_ba, 2),
    }

    print(f"\n{'Metric':<18s} {'Dir':>7s} {'Input':>10s} {'Baseline':>10s} {'Advanced':>10s} {'B→A Δ':>10s} {'Winner':>10s} {'Adv Win%':>9s}")
    print("-" * 90)

    for key, label, is_ref, higher_better in all_metrics:
        b_mean = float(np.mean(agg["baseline"][key]))
        a_mean = float(np.mean(agg["advanced"][key]))
        delta = a_mean - b_mean

        if higher_better:
            winner = "Advanced" if delta > 0 else "Baseline"
        else:
            winner = "Advanced" if delta < 0 else "Baseline"

        # Win rate
        wins = 0
        for j in range(len(selected)):
            bv, av = agg["baseline"][key][j], agg["advanced"][key][j]
            if higher_better:
                wins += 1 if av > bv else 0
            else:
                wins += 1 if av < bv else 0
        win_pct = round(100 * wins / len(selected), 1)

        direction = "↑" if higher_better else "↓"
        i_str = ""
        if not is_ref:
            i_mean = float(np.mean(agg["input"][key]))
            i_str = f"{i_mean:>10.4f}"
        else:
            i_str = f"{'—':>10s}"

        print(f"  {label:<16s} {direction:>7s} {i_str} {b_mean:>10.4f} {a_mean:>10.4f} {delta:>+10.4f} {winner:>10s} {win_pct:>8.1f}%")

        summary["metrics"][key] = {
            "baseline_mean": round(b_mean, 4),
            "advanced_mean": round(a_mean, 4),
            "delta": round(delta, 4),
            "winner": winner,
            "advanced_wins": wins,
            "advanced_win_pct": win_pct,
        }

    # FID
    fid_delta = fid_ia - fid_ib
    fid_winner = "Advanced" if fid_delta < 0 else "Baseline"
    print(f"  {'FID':<16s} {'↓':>7s} {'—':>10s} {fid_ib:>10.2f} {fid_ia:>10.2f} {fid_delta:>+10.2f} {fid_winner:>10s} {'—':>8s}")
    summary["fid"]["winner"] = fid_winner

    # Degradation distribution
    ds = agg["degradation_scores"]
    print(f"\n  Degradation Score: mean={np.mean(ds):.4f}, min={np.min(ds):.4f}, max={np.max(ds):.4f}")
    summary["degradation"] = {
        "mean": round(float(np.mean(ds)), 4),
        "min": round(float(np.min(ds)), 4),
        "max": round(float(np.max(ds)), 4),
    }

    # Overall summary
    adv_wins_count = sum(1 for m in summary["metrics"].values() if m["winner"] == "Advanced")
    total_metrics = len(summary["metrics"]) + 1  # +1 for FID
    adv_wins_count += (1 if fid_winner == "Advanced" else 0)
    print(f"\n  Advanced wins {adv_wins_count}/{total_metrics} metrics overall")

    # Save
    full = {"summary": summary, "num_images": len(selected), "per_image": per_image}
    with open(output_path, "w") as f:
        json.dump(full, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


if __name__ == "__main__":
    main()

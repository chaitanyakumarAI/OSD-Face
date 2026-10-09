"""
Advanced OSDFace Pipeline — CLI Entry Point
=============================================
Drop-in replacement for infer.py with additional controllable parameters
for the consistency-aware, adaptive restoration system.

New arguments beyond baseline:
    --guidance_scale       CFG strength (default: 1.5)
    --t_min                Min timestep for clean images (default: 200)
    --t_max                Max timestep for degraded images (default: 600)
    --fusion_strength      Override adaptive wavelet fusion (0–1, default: auto)
    --disable_adaptive_timestep   Use fixed timestep 399
    --disable_cfg                 Disable classifier-free guidance
    --disable_wavelet_fusion      Disable wavelet post-processing
"""

import os
import sys
sys.path.append(os.getcwd())

import glob
import copy
import argparse
import random

import torch
import torch.multiprocessing as mp
import torch.nn.functional as Fun
import numpy as np
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
from safetensors import safe_open
from diffusers import UNet2DConditionModel

from adaptive_pipeline import OSDFace_Advanced


# ──────────────────────────────────────────────────────────────────────── #
#  LoRA Merge (same as infer.py — duplicated here to keep infer.py clean)
# ──────────────────────────────────────────────────────────────────────── #

def merge_Unet(args):
    """Merge LoRA weights into the UNet (identical to infer.py logic)."""
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    lora_alpha = args.lora_alpha
    lora_rank = args.lora_rank
    alpha = float(lora_alpha / lora_rank)
    processed_keys = set()

    with safe_open(
        os.path.join(args.ckpt_path, "pytorch_lora_weights.safetensors"),
        framework="pt",
    ) as f:
        state_dict = {key: f.get_tensor(key) for key in f.keys()}

    state_dict_unet = unet.state_dict()

    for key in state_dict.keys():
        if "lora_A" in key:
            lora_a_key = key
            lora_b_key = key.replace("lora_A", "lora_B")
            unet_key = (
                key.replace(".lora_A.weight", ".weight").replace("unet.", "")
            )

            assert lora_b_key in state_dict and unet_key in state_dict_unet
            W_A = state_dict[lora_a_key]
            W_B = state_dict[lora_b_key]
            original_weight = state_dict_unet[unet_key]
            processed_keys.update([lora_a_key, lora_b_key])

            if (
                len(original_weight.shape) == 4
                and len(W_A.shape) == 4
                and len(W_B.shape) == 4
            ):
                out_channels, in_channels, kH, kW = original_weight.shape
                rank = W_A.shape[0]
                assert rank == lora_rank
                W_A_flat = W_A.view(rank, -1)
                W_B_flat = W_B.view(out_channels, rank)
                delta_W_flat = torch.matmul(W_B_flat, W_A_flat)
                delta_W = delta_W_flat.view(out_channels, in_channels, kH, kW)
                merged_weight = original_weight + alpha * delta_W
            else:
                merged_weight = original_weight + alpha * torch.mm(W_B, W_A)
            state_dict_unet[unet_key] = merged_weight

        elif "lora.up.weight" in key:
            lora_up_key = key
            lora_down_key = key.replace("lora.up.weight", "lora.down.weight")
            original_weight_key = (
                key.replace(".lora.up.weight", ".weight").replace("unet.", "")
            )
            assert (
                lora_down_key in state_dict
                and original_weight_key in state_dict_unet
            )
            W_up = state_dict[lora_up_key]
            W_down = state_dict[lora_down_key]
            W_orig = state_dict_unet[original_weight_key]
            processed_keys.update([lora_up_key, lora_down_key])

            if W_orig.ndim == 2:
                delta_W = torch.matmul(W_up, W_down)
                W_merged = W_orig + alpha * delta_W
            else:
                print(f"Warning: Unhandled weight shape for {original_weight_key}")
                continue
            state_dict_unet[original_weight_key] = W_merged

    remaining_lora_keys = [k for k in state_dict.keys() if k not in processed_keys]
    if remaining_lora_keys:
        print("Warning: There are unprocessed LoRA weights:")
        for key in remaining_lora_keys:
            print(f"  - {key}")
    print("Merge Done!")
    unet.load_state_dict(state_dict_unet)
    return unet


# ──────────────────────────────────────────────────────────────────────── #
#  Worker
# ──────────────────────────────────────────────────────────────────────── #

def main_worker(Unet, rank, gpu_id, image_names, weight_dtype, args):
    """Process a subset of images on a single GPU."""
    torch.cuda.set_device(gpu_id)
    model = OSDFace_Advanced(args, gpu_id, Unet).to(gpu_id)

    for image_name in tqdm(image_names, desc=f"GPU {gpu_id}"):
        output_file_path = os.path.join(
            args.output_dir, os.path.basename(image_name)
        )
        if os.path.exists(output_file_path):
            print(f"Skipping {os.path.basename(image_name)} (already exists)")
            continue

        input_image = Image.open(image_name).convert("RGB")
        with torch.no_grad():
            lq = F.to_tensor(input_image).unsqueeze(0).to(gpu_id, dtype=weight_dtype) * 2 - 1
            if lq.shape[2] == lq.shape[3]:
                lq = Fun.interpolate(lq, (512, 512), mode="bilinear", align_corners=True)

            print(f"\n{'='*60}")
            print(f"Processing: {os.path.basename(image_name)}")
            print(f"{'='*60}")

            output_image = model(lq)
            output_pil = transforms.ToPILImage()(output_image[0].cpu())
            output_pil.save(output_file_path)


def run_inference(args, Unet):
    """Multi-GPU inference dispatcher (same structure as infer.py)."""
    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f"{args.input_image}/*.[jpJP][pnPN]*[gG]"))
    else:
        image_names = [args.input_image]

    exist_images = sorted(glob.glob(f"{args.output_dir}/*.[jpJP][pnPN]*[gG]"))
    exist_image_names = [os.path.basename(img) for img in exist_images]
    image_names = [
        img for img in image_names
        if os.path.basename(img) not in exist_image_names
    ]

    print(f"Existing images: {exist_image_names}")
    random.shuffle(image_names)

    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    num_gpus = len(args.gpu_ids)
    images_per_gpu = len(image_names) // num_gpus
    processes = []

    for rank, gpu_id in enumerate(args.gpu_ids):
        start_idx = rank * images_per_gpu
        end_idx = (
            start_idx + images_per_gpu if rank != num_gpus - 1 else len(image_names)
        )
        image_subset = image_names[start_idx:end_idx]
        p = mp.Process(
            target=main_worker,
            args=(Unet, rank, gpu_id, image_subset, weight_dtype, args),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


# ──────────────────────────────────────────────────────────────────────── #
#  CLI
# ──────────────────────────────────────────────────────────────────────── #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced OSDFace Pipeline — Adaptive, Controllable Face Restoration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Baseline arguments (same as infer.py) ────────────────────────────
    parser.add_argument("--input_image", "-i", type=str, required=True,
                        help="Path to input image or directory")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="Directory to save restored outputs")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="preset/models/stable-diffusion-2-1-base",
                        help="Stable Diffusion model path")
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str,
                        choices=["fp16", "fp32"], default="fp32")
    parser.add_argument("--img_encoder_weight", type=str,
                        default="preset/pretrained/associate_2.ckpt")
    parser.add_argument("--gpu_ids", nargs="+", type=int, default=[0])
    parser.add_argument("--cat_prompt_embedding", action="store_true")
    parser.add_argument("--use_pos_embedding", action="store_true")
    parser.add_argument("--use_att_pool", action="store_true")
    parser.add_argument("--merge_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16)

    # ── Advanced pipeline arguments ──────────────────────────────────────
    parser.add_argument("--guidance_scale", type=float, default=1.5,
                        help="CFG guidance scale (1.0 = no guidance, higher = stronger conditioning)")
    parser.add_argument("--t_min", type=int, default=200,
                        help="Minimum timestep for clean images")
    parser.add_argument("--t_max", type=int, default=600,
                        help="Maximum timestep for degraded images")
    parser.add_argument("--fusion_strength", type=float, default=None,
                        help="Override adaptive wavelet fusion weight (0=all restored, 1=all original). "
                             "Default: auto-determined by degradation score.")
    parser.add_argument("--disable_adaptive_timestep", action="store_true",
                        help="Use fixed timestep (399) instead of adaptive selection")
    parser.add_argument("--disable_cfg", action="store_true",
                        help="Disable classifier-free guidance")
    parser.add_argument("--disable_wavelet_fusion", action="store_true",
                        help="Disable wavelet domain fusion post-processing")

    args = parser.parse_args()
    mp.set_start_method("spawn", force=True)

    # ── Seed ─────────────────────────────────────────────────────────────
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── LoRA Merge ───────────────────────────────────────────────────────
    Unet = None
    if args.merge_lora:
        Unet = merge_Unet(args)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Print configuration ──────────────────────────────────────────────
    print("=" * 60)
    print("  ADVANCED OSDFace PIPELINE")
    print("=" * 60)
    print(f"  Adaptive Timestep : {'ON' if not args.disable_adaptive_timestep else 'OFF'}")
    if not args.disable_adaptive_timestep:
        print(f"    └─ Range         : [{args.t_min}, {args.t_max}]")
    print(f"  CFG Guidance      : {'ON' if not args.disable_cfg else 'OFF'}")
    if not args.disable_cfg:
        print(f"    └─ Scale         : {args.guidance_scale}")
    print(f"  Wavelet Fusion    : {'ON' if not args.disable_wavelet_fusion else 'OFF'}")
    if not args.disable_wavelet_fusion:
        fs = args.fusion_strength
        print(f"    └─ Strength      : {'adaptive' if fs is None else f'{fs:.2f}'}")
    print(f"  Input              : {args.input_image}")
    print(f"  Output             : {args.output_dir}")
    print(f"  GPUs               : {args.gpu_ids}")
    print("=" * 60)

    if os.path.isdir(args.input_image):
        total = len(glob.glob(f"{args.input_image}/*"))
        print(f"There are {total} images to process.")

    run_inference(args, Unet)

    print("\n" + "=" * 60)
    print("  DONE — All images processed.")
    print("=" * 60)

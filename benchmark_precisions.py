import os
import sys
import time
import argparse
import torch
import torchvision.transforms.functional as F
from PIL import Image

sys.path.append(os.getcwd())
from infer import OSDFace_test, merge_Unet

def run_benchmark(mixed_precision):
    print(f"\n{'='*50}")
    print(f"  BENCHMARK: {mixed_precision.upper()}")
    print(f"{'='*50}")

    # Build dummy args matching test.sh
    args = argparse.Namespace(
        pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base",
        img_encoder_weight="pretrained/associate_2.ckpt",
        ckpt_path="pretrained",
        merge_lora=True,
        lora_rank=16,
        lora_alpha=16,
        mixed_precision=mixed_precision,
        cat_prompt_embedding=False,
        use_pos_embedding=False,
        use_att_pool=False,
        gpu_id=0,
        process_size=512,
        seed=114
    )

    device = torch.device(f"cuda:{args.gpu_id}")
    torch.cuda.set_device(device)
    
    # 1. Measure Peak VRAM during model loading
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    print("Merging UNet Lora...")
    Unet = merge_Unet(args)
    
    print("Loading OSDFace Model...")
    model = OSDFace_test(args, gpu_id=args.gpu_id, Unet=Unet).to(device)
    model.eval()

    vram_after_load = torch.cuda.max_memory_allocated(device) / (1024**2)
    print(f"Peak VRAM (Model Load): {vram_after_load:.2f} MB")

    # 2. Warm up
    weight_dtype = torch.float16 if mixed_precision == "fp16" else torch.float32
    dummy_input_pil = Image.fromarray(torch.randint(0, 255, (512, 512, 3), dtype=torch.uint8).numpy())

    lq = F.to_tensor(dummy_input_pil).unsqueeze(0).to(device, dtype=weight_dtype) * 2 - 1

    print("Running Warm-up...")
    with torch.no_grad():
        _ = model(lq)

    # 3. Benchmark Inference Time and Peak Inference VRAM
    torch.cuda.reset_peak_memory_stats(device)
    num_runs = 10
    total_time = 0.0

    print(f"Running Inference Benchmark ({num_runs} iters)...")
    with torch.no_grad():
        for i in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            
            _ = model(lq)
            
            torch.cuda.synchronize()
            end = time.time()
            total_time += (end - start)

    avg_time = (total_time / num_runs) * 1000  # ms
    peak_vram_inf = torch.cuda.max_memory_allocated(device) / (1024**2)

    print(f"Avg Inference Time per image: {avg_time:.2f} ms")
    print(f"Peak VRAM (During Inference): {peak_vram_inf:.2f} MB")

    return {
        "precision": mixed_precision,
        "load_vram": vram_after_load,
        "inf_vram": peak_vram_inf,
        "avg_time_ms": avg_time
    }

if __name__ == "__main__":
    fp32_res = run_benchmark("fp32")
    
    # Clean up completely to test fp16
    torch.cuda.empty_cache()
    
    fp16_res = run_benchmark("fp16")

    print("\n\n" + "="*50)
    print("                 SUMMARY ")
    print("="*50)
    print(f"{'Metric':<25} | {'FP32':<10} | {'FP16':<10}")
    print("-" * 50)
    print(f"{'Avg Inference Time':<25} | {fp32_res['avg_time_ms']:>6.2f} ms  | {fp16_res['avg_time_ms']:>6.2f} ms")
    print(f"{'Peak VRAM (Inference)':<25} | {fp32_res['inf_vram']:>6.2f} MB  | {fp16_res['inf_vram']:>6.2f} MB")
    print(f"{'Peak VRAM (Model Load)':<25} | {fp32_res['load_vram']:>6.2f} MB  | {fp16_res['load_vram']:>6.2f} MB")
    print("="*50)

    # Save to a text file
    with open("precision_benchmark_results.txt", "w") as f:
        f.write("OSDFace Precision Benchmark (FP32 vs FP16)\n")
        f.write("="*50 + "\n")
        f.write(f"{'Metric':<25} | {'FP32':<10} | {'FP16':<10}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Avg Inference Time':<25} | {fp32_res['avg_time_ms']:>6.2f} ms  | {fp16_res['avg_time_ms']:>6.2f} ms\n")
        f.write(f"{'Peak VRAM (Inference)':<25} | {fp32_res['inf_vram']:>6.2f} MB  | {fp16_res['inf_vram']:>6.2f} MB\n")
        f.write(f"{'Peak VRAM (Model Load)':<25} | {fp32_res['load_vram']:>6.2f} MB  | {fp16_res['load_vram']:>6.2f} MB\n")
        f.write("="*50 + "\n")

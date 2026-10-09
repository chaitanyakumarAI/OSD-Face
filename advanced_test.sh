#!/bin/bash
# ============================================================
#  Advanced OSDFace Pipeline — Test Script
#  Runs the adaptive, controllable face restoration system
#  on the WebPhoto-Test dataset.
# ============================================================

ckpt_path="pretrained"

echo "============================================================"
echo "  Running Advanced OSDFace Pipeline"
echo "============================================================"

python run_advanced_pipeline.py \
    --input_image data/WebPhoto-Test \
    --output_dir results/Advanced-WebPhoto-Test \
    --pretrained_model_name_or_path "stabilityai/stable-diffusion-2-1-base" \
    --img_encoder_weight "$ckpt_path/associate_2.ckpt" \
    --gpu_ids 0 \
    --ckpt_path $ckpt_path \
    --merge_lora \
    --guidance_scale 1.5 \
    --t_min 200 \
    --t_max 600

echo ""
echo "============================================================"
echo "  Done! Results saved to: results/Advanced-WebPhoto-Test/"
echo "============================================================"

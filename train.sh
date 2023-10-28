export CUDA_VISIBLE_DEVICES="0,1";
accelerate launch --mixed_precision=fp16 --multi_gpu train.py \
    --dset imagenet_syn \
    --train_dir datasets/imagenet_syn_clip_prompts \
    --val_dir datasets/val \
    --test_dir datasets/val \
    --model resnet50 \
    --num-classes 100 \
    --output_dir output/clip_prompts \
    -b 128

export CUDA_VISIBLE_DEVICES="0,1,2,3";
accelerate launch --mixed_precision=bf16 --multi_gpu train.py \
    --train_dir datasets/imagenet_syn_100img100cls_no_noise \
    --val_dir datasets/val \
    --test_dir datasets/val \
    --model resnet50 \
    --aa v0 \
    --mixup 0.2 \
    -b 128

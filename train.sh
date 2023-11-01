export CUDA_VISIBLE_DEVICES="0";
accelerate launch --mixed_precision=fp16 --multi_gpu train.py \
    --dset imagenet_syn \
    --train_dir /lab/tmpig7b/u/brian-data/imagenet_syn_100img_reparam_ortho \
    --val_dir /lab/tmpig7b/u/brian-data/imagenet/images/val \
    --test_dir /lab/tmpig7b/u/brian-data/imagenet/images/val \
    --model resnet50 \
    --num-classes 100 \
    --aa v0 \
    --mixup 0.2 \
    --epochs 300 \
    --output_dir output/100img_reparam_ortho \
    -b 128 

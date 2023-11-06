export CUDA_VISIBLE_DEVICES="0,1";
accelerate launch --mixed_precision=fp16 --multi_gpu train.py \
    --dset imagenet_syn \
    --train_dir /lab/tmpig7b/u/brian-data/imagenet_syn_fake_it \
    --val_dir /lab/tmpig7b/u/brian-data/imagenet/images/val \
    --test_dir /lab/tmpig7b/u/brian-data/imagenet/images/val \
    --model resnet50 \
    --num-classes 100 \
    --aa v0 \
    --mixup 0.2 \
    --epochs 300 \
    --output_dir output/fake_it \
    -b 256

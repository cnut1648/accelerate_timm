export CUDA_VISIBLE_DEVICES="0,1";
accelerate launch --mixed_precision=fp16 --multi_gpu train.py \
    --dset imagenet_syn \
    --train_dir /lab/tmpig7b/u/brian-data/dreambooth_medium \
    --val_dir /lab/tmpig7b/u/brian-data/imagenet/images/val \
    --test_dir /lab/tmpig7b/u/brian-data/imagenet/images/val \
    --portions -1 \
    --model resnet50 \
    --num-classes 100 \
    --aa v0 \
    --mixup 0.2 \
    --epochs 200 \
    --output_dir output/dreambooth_medium \
    -b 256

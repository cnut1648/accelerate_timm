REAL_DATA_DIR="/lab/tmpig7b/u/brian-data/imagenet/images/train"
SYN_DATA_DIR="/lab/tmpig7b/u/brian-data/imagenet_syn_1300img_reparam_ortho"
VAL_DATA_DIR="/lab/tmpig7b/u/brian-data/imagenet/images/val"
OUTPUT_DIR="tmp"
SYN=0.2


accelerate launch --mixed_precision=fp16 --multi_gpu train.py \
    --dset imagenet_real imagenet_syn \
    --train_dir "$REAL_DATA_DIR" "$SYN_DATA_DIR" \
    --portions 1 $SYN \
    --val_dir "$VAL_DATA_DIR" \
    --test_dir "$VAL_DATA_DIR" \
    --model resnet50 \
    --num-classes 100 \
    --aa v0 \
    --mixup 0.2 \
    --epochs 300 \
    --output_dir "$OUTPUT_DIR" \
    -b 256

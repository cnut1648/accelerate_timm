# python train.py --train_dir /home/ec2-user/accelerate_timm/imagenet_syn_100img100cls_no_noise \
#     --val_dir /home/ec2-user/accelerate_timm/val \
#     --test_dir /home/ec2-user/accelerate_timm/val \
#     --model resnet50 \
#     --aa v0 --mixup 0.2
# accelerate launch --multi_gpu --mixed_precision=bf16 --gpu_ids="0,1" train.py --train_dir /home/ec2-user/accelerate_timm/imagenet_syn_100img100cls_no_noise \
#     --val_dir /home/ec2-user/accelerate_timm/val \
#     --test_dir /home/ec2-user/accelerate_timm/val \
#     --model resnet50 \
#     --aa v0 --mixup 0.2 -b 2048
export CUDA_VISIBLE_DEVICES="0,1,2,3";
accelerate launch --mixed_precision=bf16 --multi_gpu train.py \
    --train_dir /home/ec2-user/accelerate_timm/imagenet_syn_100img100cls_no_noise \
    --val_dir /home/ec2-user/accelerate_timm/val \
    --test_dir /home/ec2-user/accelerate_timm/val \
    --model resnet50 \
    --aa v0 --mixup 0.2 -b 512
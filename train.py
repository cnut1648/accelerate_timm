# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import re
import random
from datetime import datetime

import numpy as np
import PIL
import torch
import timm
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.utils.metrics import accuracy
from einops import rearrange, repeat

import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger

from custom_dataset import ImageNet100, SynImageFolder, add_dataset_args, get_train_val_test_dataset
from dino_augmentation import DataAugmentationDINO

########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a ResNet50 on the Oxford-IIT Pet Dataset
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################


logger = get_logger(__name__)

def training_function(args):
    # Initialize accelerator
    if args.with_tracking:
        accelerator = Accelerator(
            cpu=args.cpu, mixed_precision=args.mixed_precision, log_with="wandb", project_dir=args.output_dir,
            gradient_accumulation_steps=args.grad_accum_steps)
    else:
        accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision,
            gradient_accumulation_steps=args.grad_accum_steps)

    # Parse out whether we are saving every epoch or after a certain number of batches
    if hasattr(args.checkpointing_steps, "isdigit"):
        if args.checkpointing_steps == "epoch":
            checkpointing_steps = args.checkpointing_steps
        elif args.checkpointing_steps.isdigit():
            checkpointing_steps = int(args.checkpointing_steps)
        else:
            raise ValueError(
                f"Argument `checkpointing_steps` must be either a number or `epoch`. `{args.checkpointing_steps}` passed."
            )
    else:
        checkpointing_steps = None

    # We need to initialize the trackers we use, and also store our configuration
    if args.with_tracking:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run, vars(args))
    os.makedirs(args.output_dir, exist_ok=True)
    # write each k v of args into a text file in output_dir
    with open(os.path.join(args.output_dir, 'args.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    if args.fast_norm:
        timm.layers.set_fast_norm()

    # Set the seed before splitting the data.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    accelerate.utils.set_seed(args.seed)
    
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]
    model = timm.create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        checkpoint_path=args.initial_checkpoint,
        **args.model_kwargs,
    )
    if args.head_init_scale is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(args.head_init_scale)
            model.get_classifier().bias.mul_(args.head_init_scale)
    if args.head_init_bias is not None:
        torch.nn.init.constant_(model.get_classifier().bias, args.head_init_bias)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    # Freezing the base model
    if args.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.get_classifier().parameters():
            param.requires_grad = True

    if args.torchcompile:
        # NOTE torch compile should be done after DDP
        model = torch.compile(model, backend=args.torchcompile)
    
    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = timm.utils.ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)

    data_config = timm.data.resolve_data_config(vars(args), model=model, verbose=accelerator.is_main_process)
    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    train_dataset, val_dataset, test_dataset = get_train_val_test_dataset(args.dset, args.train_dirs, args.val_dir, args.test_dir)
    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )
        if not args.no_prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = timm.data.FastCollateMixup(**mixup_args)
        else:
            mixup_fn = timm.data.Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        dataset_train = timm.data.AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
    train_dataloader = timm.data.create_loader(
        train_dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=not args.no_prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=accelerator.use_distributed,
        collate_fn=collate_fn,
        pin_memory=True,
        device=accelerator.device,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding="all",
    )
    val_dataloader = timm.data.create_loader(
        val_dataset,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=not args.no_prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=accelerator.use_distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=True,
        device=accelerator.device,
    )
    test_dataloader = timm.data.create_loader(
        test_dataset,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=not args.no_prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        distributed=accelerator.use_distributed,
        crop_pct=data_config['crop_pct'],
        pin_memory=True,
        device=accelerator.device,
    )

    if args.dino_aug:
        # Default values from https://github.com/facebookresearch/dino/blob/main/main_dino.py
        train_dataloader.dataset.transform = DataAugmentationDINO(
            global_crops_scale=(0.4, 1),
            local_crops_scale=(0.05, 4),
            local_crops_number=8,
        )

    # We could avoid this line since the accelerator is set with `device_placement=True` (default value).
    # Note that if you are placing tensors on devices manually, this line absolutely needs to be before the optimizer
    # creation otherwise training will not work on TPU (`accelerate` will kindly throw an error to make us aware of that).
    # model = model.to(accelerator.device) # TODO
        
    if not args.lr:
        global_batch_size = args.batch_size * accelerator.num_processes * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        logger.info(
            f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) '
            f'and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.', main_process_only=True)

    # Instantiate optimizer
    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )

    # setup learning rate schedule and starting epoch
    updates_per_epoch = (len(train_dataloader) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )

    logger.info(
        f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.', main_process_only=True)

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, val_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, test_dataloader, lr_scheduler
    )
    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = torch.nn.CrossEntropyLoss()
    # train_loss_fn = train_loss_fn.to(device=accelerator.device)
    # validate_loss_fn = torch.nn.CrossEntropyLoss().to(device=accelerator.device)
    validate_loss_fn = torch.nn.CrossEntropyLoss()

    # We need to keep track of how many total steps we have iterated over
    overall_step = 0
    # We also need to keep track of the starting epoch so files are named properly
    starting_epoch = 0

    # Potentially load in the weights only from a previous save
    if args.load_checkpoint is not None:
        accelerator.print(f"Load checkpoint: {args.load_checkpoint}")
        model.load_state_dict(torch.load(args.load_checkpoint))
        #accelerate.load_checkpoint_and_dispatch(model, args.load_checkpoint)
        args.resume_from_checkpoint = None

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            try:
                accelerator.load_state(args.resume_from_checkpoint)
            except:
                accelerate.load_checkpoint_and_dispatch(model, args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
            if args.model_ema:
                timm.models.load_checkpoint(model_ema.module, args.resume, use_ema=True)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            resume_step = int(training_difference.replace("step_", ""))
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    if lr_scheduler is not None and starting_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(starting_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(starting_epoch)

    # Now we train the model
    accelerator.wait_for_everyone()
    for epoch in range(starting_epoch, num_epochs):
        model.train()
        if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
            if not args.no_prefetcher and train_dataloader.mixup_enabled:
                train_dataloader.mixup_enabled = False
            elif mixup_fn is not None:
                mixup_fn.mixup_enabled = False
        
        optimizer.zero_grad(set_to_none=True)
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We need to skip steps until we reach the resumed step
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            overall_step += resume_step
        else:
            # After the first iteration though, we need to go back to the original dataloader
            active_dataloader = train_dataloader
        
        train_one_epoch(
            args, model, optimizer, lr_scheduler, train_loss_fn, active_dataloader, accelerator,
            epoch, overall_step, checkpointing_steps, model_ema=model_ema, mixup_fn=mixup_fn
        )
        accelerator.wait_for_everyone() 
        eval_metric = evaluate(args, model, validate_loss_fn, val_dataloader, accelerator, epoch, checkpointing_steps)
        # Use accelerator.print to print only on the main process.
        accelerator.print(f"epoch {epoch}: top1={eval_metric[0]:.2f}, top5={eval_metric[1]:.2f}")

    test_metric = evaluate(args, model, validate_loss_fn, test_dataloader, accelerator, epoch, checkpointing_steps)
    accelerator.print(f"test: top1={test_metric[0]:.2f}, top5={test_metric[1]:.2f}")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        output_dir = "final_weights.pt"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
        torch.save(model.state_dict(), output_dir)
        print(f"model saved at {os.path.join(os.getcwd(), output_dir)}") 

    if args.with_tracking:
        accelerator.end_training()

def train_one_epoch(
    args, model, optimizer, lr_scheduler, train_loss_fn, active_dataloader, accelerator, 
    epoch, overall_step, checkpointing_steps, model_ema=None, mixup_fn=None
):
    total_loss = 0
    for batch_idx, (inputs, targets) in enumerate(active_dataloader):
        with accelerator.accumulate(model):
            if args.dino_aug:
                assert isinstance(active_dataloader.dataset.transform, DataAugmentationDINO)
                assert inputs.ndim == 5
                l = inputs.shape[1]
                inputs = rearrange(inputs, "b l c h w -> (b l) c h w")
                targets = repeat(targets, "b -> (b l)", l=l)

            if mixup_fn is not None:
                inputs, targets = mixup_fn(inputs, targets)
            # inputs, targets = inputs.to(accelerator.device), targets.to(accelerator.device)

            # Debug: show transformed image
            # from torchvision import transforms
            # mean = torch.squeeze(active_dataloader.mean, dim=0)
            # std = torch.squeeze(active_dataloader.std, dim=0)
            # (transforms.ToPILImage(mode="RGB")((inputs[0] * std + mean).to(torch.int8))).save("tmp0.png")

            outputs = model(inputs)
            loss = train_loss_fn(outputs, targets)
            # We keep track of the loss at each epoch
            total_loss += loss.detach().float()
            accelerator.backward(loss)
        
            optimizer.step()
            if model_ema is not None:
                model_ema.update(model)
            optimizer.zero_grad()
            overall_step += 1
            if overall_step % 1000 == 0:
                accelerator.print(
                    f'Train: {epoch} {batch_idx}/{len(active_dataloader)}({batch_idx/len(active_dataloader)*100 :>3.0f}%) Loss: {loss.item()}',
                    # main_process_only=True,
                )
            if isinstance(checkpointing_steps, int):
                output_dir = f"step_{overall_step}"
                if overall_step % checkpointing_steps == 0:
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
    avg_loss = total_loss.item() / len(active_dataloader)
    if args.with_tracking:
        accelerator.log(
            {
                "train_loss": avg_loss,
                "epoch": epoch,
            },
            step=overall_step,
        )
    
    if lr_scheduler is not None:
        lr_scheduler.step(epoch, metric=avg_loss)

def evaluate(
    args, model, validate_loss_fn, val_dataloader, accelerator, epoch, checkpointing_steps
):
    model.eval()
    accurate = 0
    num_elems = 0
    total_loss = 0
    logits, refs = torch.Tensor([]).to(model.device), torch.Tensor([]).to(model.device)
    for step, (inputs, targets) in enumerate(val_dataloader):
        with torch.no_grad():
            outputs = model(inputs)
        # augmentation reduction
        reduce_factor = args.tta
        if reduce_factor > 1:
            outputs = outputs.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
            targets = targets[0:targets.size(0):reduce_factor]

        loss = validate_loss_fn(outputs, targets)
        logit, ref = accelerator.gather_for_metrics((outputs, targets))
        logits = torch.concat([logits, logit], dim=0)
        refs = torch.concat([refs, ref], dim=0)
        prediction = logit.argmax(dim=-1)
        accurate_preds = prediction == ref
        num_elems += accurate_preds.shape[0]
        accurate += accurate_preds.long().sum()
    eval_metric = accuracy(logits, refs, topk=(1,5))
    #print(100*accurate.item() / num_elems)
    if checkpointing_steps == "epoch":
        output_dir = f"epoch_{epoch}"
        if args.output_dir is not None:
            output_dir = os.path.join(args.output_dir, output_dir)
        accelerator.save_state(output_dir)
    
    if args.with_tracking:
        accelerator.log(
            {
                "acc": eval_metric,
                "epoch": epoch,
            },
        )
    
    return eval_metric

def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser = add_dataset_args(parser)
    group = parser.add_argument_group('I/O')
    group.add_argument(
        "--output_dir", type=str,
        default=f"output/{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
        help="path to output directory [default: output/year-month-date_hour-minute]",
    )
    group.add_argument("--cluster", action="store_true", help="run on cluster")
    group.add_argument("--workers", type=int, default=4, help="number of workers for dataloader")

    group = parser.add_argument_group('MODEL')
    group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                       help='Name of model to train (default: "resnet50")')
    group.add_argument('--pretrained', action='store_true', default=False,
                       help='Start with pretrained version of specified network (if avail)')
    group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                       help='Initialize model from this checkpoint (default: none)')
    group.add_argument('--num-classes', type=int, default=None, metavar='N',
                       help='number of label classes (Model default if None)')
    group.add_argument('--gp', default=None, type=str, metavar='POOL',
                       help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    group.add_argument('--in-chans', type=int, default=None, metavar='N',
                       help='Image input channels (default: None => 3)')
    group.add_argument('--input-size', default=None, nargs=3, type=int,
                       metavar='N N N',
                       help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    group.add_argument('--crop-pct', default=None, type=float,
                       metavar='N', help='Input image center crop percent (for validation only)')
    group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                       help='Override mean pixel value of dataset')
    group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                       help='Override std deviation of dataset')
    group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                       help='Image resize interpolation type (overrides model)')
    group.add_argument('--fast-norm', default=False, action='store_true',
                       help='enable experimental fast-norm')
    group.add_argument('--model-kwargs', nargs='*', default={}, action=timm.utils.ParseKwargs)
    group.add_argument('--head-init-scale', default=None, type=float,
                       help='Head initialization scale')
    group.add_argument('--head-init-bias', default=None, type=float,
                       help='Head initialization bias value')
    group.add_argument("--freeze_backbone", action="store_true", help="If passed, will freeze the backbone and only train head.")

    group = parser.add_argument_group('OPTIM')
    group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                       help='Optimizer (default: "sgd")')
    group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                       help='Optimizer Epsilon (default: None, use opt default)')
    group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                       help='Optimizer Betas (default: None, use opt default)')
    group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                       help='Optimizer momentum (default: 0.9)')
    group.add_argument('--weight-decay', type=float, default=2e-5,
                       help='weight decay (default: 2e-5)')
    group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                       help='Clip gradient norm (default: None, no clipping)')
    group.add_argument('--clip-mode', type=str, default='norm',
                       help='Gradient clipping mode. One of ("norm", "value", "agc")')
    group.add_argument('--layer-decay', type=float, default=None,
                       help='layer-wise learning rate decay (default: None)')
    group.add_argument('--opt-kwargs', nargs='*', default={}, action=timm.utils.ParseKwargs)
    

    group = parser.add_argument_group('LR')
    group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                       help='LR scheduler (default: "step"')
    group.add_argument('--sched-on-updates', action='store_true', default=False,
                       help='Apply LR scheduler step on update instead of epoch end.')
    group.add_argument('--lr', type=float, default=None, metavar='LR',
                       help='learning rate, overrides lr-base if set (default: None)')
    group.add_argument('--lr-base', type=float, default=0.1, metavar='LR',
                       help='base learning rate: lr = lr_base * global_batch_size / base_size')
    group.add_argument('--lr-base-size', type=int, default=256, metavar='DIV',
                       help='base learning rate batch size (divisor, default: 256).')
    group.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                       help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
    group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                       help='learning rate noise on/off epoch percentages')
    group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                       help='learning rate noise limit percent (default: 0.67)')
    group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                       help='learning rate noise std-dev (default: 1.0)')
    group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                       help='learning rate cycle len multiplier (default: 1.0)')
    group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                       help='amount to decay each learning rate cycle (default: 0.5)')
    group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                       help='learning rate cycle limit, cycles enabled if > 1')
    group.add_argument('--lr-k-decay', type=float, default=1.0,
                       help='learning rate k-decay for cosine/poly (default: 1.0)')
    group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                       help='warmup learning rate (default: 1e-5)')
    group.add_argument('--min-lr', type=float, default=0, metavar='LR',
                       help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
    group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                       help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                       help='manual epoch number (useful on restarts)')
    group.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES",
                       help='list of decay epoch indices for multistep lr. must be increasing')
    group.add_argument('--decay-epochs', type=float, default=90, metavar='N',
                       help='epoch interval to decay LR')
    group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                       help='epochs to warmup LR, if scheduler supports')
    group.add_argument('--warmup-prefix', action='store_true', default=False,
                       help='Exclude warmup period from decay schedule.'),
    group.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                       help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                       help='patience epochs for Plateau LR scheduler (default: 10)')
    group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                       help='LR decay rate (default: 0.1)')
                       
    group = parser.add_argument_group('AUGMENTATION')
    group.add_argument('--no-aug', action='store_true', default=False,
                       help='Disable all training augmentation, override other train aug args')
    group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                       help='Random resize scale (default: 0.08 1.0)')
    group.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                       help='Random resize aspect ratio (default: 0.75 1.33)')
    group.add_argument('--hflip', type=float, default=0.5,
                       help='Horizontal flip training aug probability')
    group.add_argument('--vflip', type=float, default=0.,
                       help='Vertical flip training aug probability')
    group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                       help='Color jitter factor (default: 0.4)')
    group.add_argument('--aa', type=str, default=None, metavar='NAME',
                       help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    group.add_argument('--aug-repeats', type=float, default=0,
                       help='Number of augmentation repetitions (distributed training only) (default: 0)')
    group.add_argument('--aug-splits', type=int, default=0,
                       help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    group.add_argument('--jsd-loss', action='store_true', default=False,
                       help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    group.add_argument('--bce-loss', action='store_true', default=False,
                       help='Enable BCE loss w/ Mixup/CutMix use.')
    group.add_argument('--bce-target-thresh', type=float, default=None,
                       help='Threshold for binarizing softened BCE targets (default: None, disabled)')
    group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                       help='Random erase prob (default: 0.)')
    group.add_argument('--remode', type=str, default='pixel',
                       help='Random erase mode (default: "pixel")')
    group.add_argument('--recount', type=int, default=1,
                       help='Random erase count (default: 1)')
    group.add_argument('--resplit', action='store_true', default=False,
                       help='Do not random erase first (clean) augmentation split')
    group.add_argument('--mixup', type=float, default=0.0,
                       help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    group.add_argument('--cutmix', type=float, default=0.0,
                       help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                       help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    group.add_argument('--mixup-prob', type=float, default=1.0,
                       help='Probability of performing mixup or cutmix when either/both is enabled')
    group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                       help='Probability of switching to cutmix when both mixup and cutmix enabled')
    group.add_argument('--mixup-mode', type=str, default='batch',
                       help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                       help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    group.add_argument('--smoothing', type=float, default=0.1,
                       help='Label smoothing (default: 0.1)')
    group.add_argument('--train-interpolation', type=str, default='random',
                       help='Training interpolation (random, bilinear, bicubic default: "random")')
    group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                       help='Dropout rate (default: 0.)')
    group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                       help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                       help='Drop path rate (default: None)')
    group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                       help='Drop block rate (default: None)')
    group.add_argument("--dino-aug", action='store_true', default=False,
                       help='Use DINO multi-crop augmentation (default: None)')

    group = parser.add_argument_group('EMA')
    group.add_argument('--model-ema', action='store_true', default=False,
                       help='Enable tracking moving average of model weights')
    group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                       help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    group.add_argument('--model-ema-decay', type=float, default=0.9998,
                       help='decay factor for model weights moving average (default: 0.9998)')

    group = parser.add_argument_group('TRAINING')
    group.add_argument('--no-prefetcher', action='store_true', default=False,
                   help='disable fast prefetcher')
    group.add_argument('--grad-accum-steps', type=int, default=1, metavar='N',
                       help='The number of steps to accumulate gradients (default: 1)')
    group.add_argument('--grad-checkpointing', action='store_true', default=False,
                       help='Enable gradient checkpointing through model blocks/stages')
    group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                       help='Input batch size for training (default: 128)')
    group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                       help='Validation batch size override (default: None)')
    group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
    group.add_argument("--fp16", action="store_true", help="If passed, will use FP16 training.")
    group.add_argument('--epochs', type=int, default=300, metavar='N',
                       help='number of epochs to train (default: 300)')
    group.add_argument("--seed", default=42, type=int, help="Seed for the random number generator.")
    group.add_argument('--tta', type=int, default=0, metavar='N',
                   help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
    group.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    group.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    group.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    group.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a saved weights checkpoint.",
    )
    group.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    group.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                   help='use the multi-epochs-loader to save time at the beginning of every epoch')
    args = parser.parse_args()

    if args.cluster:
        args.data_root = os.path.join(os.environ['PT_DATA_DIR'], args.data_root)
        args.out_dir = os.path.join(os.environ['PT_DATA_DIR'], args.out_dir)

    training_function(args)


if __name__ == "__main__":
    main()

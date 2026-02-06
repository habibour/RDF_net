"""
RDFNet Training Script for Google Colab with Resume Support
"""

import datetime
import os
import shutil
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.model import YoloBody
from nets.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import (get_anchors, get_classes,
                         seed_everything, show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch
from colab_config import *


def find_latest_checkpoint(checkpoint_dir, local_logs='logs'):
    """Find the latest checkpoint from Drive or local logs"""
    latest = None
    latest_epoch = -1
    
    for loc in [local_logs, checkpoint_dir]:
        if os.path.exists(loc):
            for f in os.listdir(loc):
                if f.startswith('ep') and f.endswith('.pth'):
                    try:
                        epoch = int(f.split('-')[0][2:])
                        if epoch > latest_epoch:
                            latest_epoch = epoch
                            latest = os.path.join(loc, f)
                    except:
                        pass
    
    return latest, latest_epoch


def backup_checkpoint(src_dir, dst_dir):
    """Backup checkpoints to Drive"""
    if not os.path.exists(src_dir):
        return
    
    os.makedirs(dst_dir, exist_ok=True)
    
    for f in os.listdir(src_dir):
        if f.endswith('.pth'):
            src = os.path.join(src_dir, f)
            dst = os.path.join(dst_dir, f)
            if not os.path.exists(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
                shutil.copy2(src, dst)
                print(f"💾 Backed up: {f}")


def restore_checkpoints(src_dir, dst_dir):
    """Restore checkpoints from Drive"""
    if not os.path.exists(src_dir):
        return
    
    os.makedirs(dst_dir, exist_ok=True)
    
    for f in os.listdir(src_dir):
        if f.endswith('.pth'):
            src = os.path.join(src_dir, f)
            dst = os.path.join(dst_dir, f)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
                print(f"📥 Restored: {f}")


if __name__ == "__main__":
    print("=" * 60)
    print("RDFNet Training with Resume Support")
    print("=" * 60)
    
    # Setup
    seed_everything(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Restore any existing checkpoints from Drive
    restore_checkpoints(CHECKPOINT_DIR, save_dir)
    
    # Check for resume
    resume_checkpoint, resume_epoch = find_latest_checkpoint(CHECKPOINT_DIR, save_dir)
    
    if resume_checkpoint and resume_epoch >= 0:
        print(f"\n📌 Found checkpoint: {os.path.basename(resume_checkpoint)}")
        print(f"📌 Resuming from epoch: {resume_epoch}")
        current_model_path = resume_checkpoint
        current_init_epoch = resume_epoch
        current_freeze_train = resume_epoch < Freeze_Epoch
    else:
        print(f"\n📭 No checkpoint found. Starting fresh training.")
        current_model_path = model_path
        current_init_epoch = Init_Epoch
        current_freeze_train = Freeze_Train
    
    # Load classes and anchors
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)
    
    print(f"\n📊 Classes: {num_classes}")
    print(f"📊 Anchors: {num_anchors}")
    
    # Build model
    model = YoloBody(anchors_mask, num_classes)
    
    if not pretrained:
        weights_init(model)
    
    # Load weights
    if current_model_path != '':
        print(f'\n📦 Loading weights: {current_model_path}')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(current_model_path, map_location=device)
        
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        
        print(f"✅ Loaded {len(load_key)} keys")
        if no_load_key:
            print(f"⚠️ Skipped {len(no_load_key)} keys")
    
    # Loss function
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, anchors_mask)
    
    # Loss history
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    
    # Mixed precision
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None
    
    model_train = model.train()
    
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    
    ema = ModelEMA(model_train)
    
    # Load dataset
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    
    num_train = len(train_lines)
    num_val = len(val_lines)
    
    print(f"\n📊 Training samples: {num_train}")
    print(f"📊 Validation samples: {num_val}")
    
    # Training config
    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        wanted_epoch = wanted_step * Unfreeze_batch_size // num_train + 1
        print(f"\n⚠️ Recommended epochs: {wanted_epoch}")
    
    # Show config
    show_config(
        classes_path=classes_path, anchors_path=anchors_path, anchors_mask=anchors_mask,
        model_path=current_model_path, input_shape=input_shape,
        Init_Epoch=current_init_epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
        Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size,
        Freeze_Train=current_freeze_train,
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
        lr_decay_type=lr_decay_type,
        save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
    )
    
    UnFreeze_flag = False
    
    if current_freeze_train:
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    batch_size = Freeze_batch_size if current_freeze_train else Unfreeze_batch_size
    nbs = 64
    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    
    # Optimizer
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)
    
    optimizer = {
        'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
        'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
    }[optimizer_type]
    
    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg2})
    
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
    
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("Dataset too small!")
    
    train_dataset = YoloDataset(train_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, train=True)
    val_dataset = YoloDataset(val_lines, input_shape, num_classes, epoch_length=UnFreeze_Epoch, train=False)
    
    train_sampler = None
    val_sampler = None
    shuffle = True
    
    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                     pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate,
                     sampler=train_sampler)
    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate,
                         sampler=val_sampler)
    
    eval_callback = EvalCallback(model, input_shape, anchors, anchors_mask, class_names, num_classes,
                                 val_lines, log_dir, Cuda, eval_flag=eval_flag, period=eval_period)
    
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    try:
        for epoch in range(current_init_epoch, UnFreeze_Epoch):
            # Unfreeze backbone
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                
                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                
                for param in model.backbone.parameters():
                    param.requires_grad = True
                
                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size
                
                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("Dataset too small!")
                
                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate,
                                 sampler=train_sampler)
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate,
                                     sampler=val_sampler)
                
                UnFreeze_flag = True
                print(f"\n🔓 Unfreezing backbone at epoch {epoch}")
            
            train_dataset.epoch_now = epoch
            val_dataset.epoch_now = epoch
            
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            
            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, fp16, scaler,
                          save_period, save_dir, local_rank=0)
            
            # Backup to Drive after each epoch
            if (epoch + 1) % save_period == 0:
                backup_checkpoint(save_dir, CHECKPOINT_DIR)
    
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted!")
    
    finally:
        # Final backup
        backup_checkpoint(save_dir, CHECKPOINT_DIR)
        print("\n" + "=" * 60)
        print("✅ Training complete!")
        print(f"📁 Checkpoints saved to: {CHECKPOINT_DIR}")
        print("=" * 60)

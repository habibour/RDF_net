"""
RDFNet Kaggle Training Script with Resume Support
"""
import datetime
import os
import shutil
from functools import partial

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

# ============= KAGGLE CONFIG =============
# Dataset paths (read-only)
VOC2007_FOG = '/kaggle/input/foggy-voc/VOC_FOG_12K_Upload/VOC_FOG_12K_Upload/VOC2007_FOG'
VOC2007_ANN = '/kaggle/input/foggy-voc/VOC_FOG_12K_Upload/VOC_FOG_12K_Upload/VOC2007_Annotations'
VOC2012_FOG = '/kaggle/input/foggy-voc/VOC_FOG_12K_Upload/VOC_FOG_12K_Upload/VOC2012_FOG'
VOC2012_ANN = '/kaggle/input/foggy-voc/VOC_FOG_12K_Upload/VOC_FOG_12K_Upload/VOC2012_Annotations'

# Output directory (writable)
OUTPUT_DIR = '/kaggle/working'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

# Classes
VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
               "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Model config
seed = 114514
fp16 = True
model_path = 'model_data/yolov7_tiny_weights.pth'
classes_path = 'model_data/voc_classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
input_shape = [640, 640]

# Training config
Freeze_Epoch = 100
Freeze_batch_size = 16
UnFreeze_Epoch = 300
Unfreeze_batch_size = 8
Freeze_Train = True

# Optimizer
Init_lr = 1e-2
Min_lr = Init_lr * 0.01
optimizer_type = "sgd"
momentum = 0.937
weight_decay = 5e-4
lr_decay_type = "cos"

# Save config
save_period = 5  # Save every 5 epochs for Kaggle
save_dir = os.path.join(OUTPUT_DIR, 'logs')
eval_flag = True
eval_period = 10
num_workers = 2  # Kaggle has limited workers

# Annotation paths
train_annotation_path = os.path.join(OUTPUT_DIR, 'train_12k.txt')
val_annotation_path = os.path.join(OUTPUT_DIR, 'val_12k.txt')


def find_latest_checkpoint():
    """Find the latest checkpoint"""
    checkpoints = []
    
    for search_dir in [save_dir, CHECKPOINT_DIR]:
        if not os.path.exists(search_dir):
            continue
        for f in os.listdir(search_dir):
            if f.startswith('ep') and f.endswith('.pth'):
                try:
                    epoch = int(f.split('-')[0][2:])
                    checkpoints.append((epoch, os.path.join(search_dir, f)))
                except:
                    pass
            elif f == 'last_epoch_weights.pth':
                # Try to determine epoch from training state
                checkpoints.append((0, os.path.join(search_dir, f)))
    
    if checkpoints:
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        return checkpoints[0]
    return None, None


def backup_checkpoint(local_path):
    """Backup checkpoint to checkpoint directory"""
    if not os.path.exists(local_path):
        return
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    filename = os.path.basename(local_path)
    shutil.copy2(local_path, os.path.join(CHECKPOINT_DIR, filename))
    print(f"💾 Backed up: {filename}")


if __name__ == "__main__":
    
    seed_everything(seed)
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda' if cuda_available else 'cpu')
    
    print("=" * 60)
    print("🚀 RDFNet Training (Kaggle)")
    print("=" * 60)
    print(f"Device: {device}")
    
    if not cuda_available:
        print("⚠️ GPU NOT AVAILABLE!")
        print("⚠️ Enable GPU: Settings → Accelerator → GPU")
        raise RuntimeError("GPU required for training")
    
    # Get classes and anchors
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)
    
    print(f"📊 Classes: {num_classes}")
    print(f"📊 Anchors: {len(anchors)}")
    
    # Create model
    model = YoloBody(anchors_mask, num_classes)
    weights_init(model)
    
    # Setup directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Check for resume
    resume_epoch = 0
    latest_epoch, latest_ckpt = find_latest_checkpoint()
    
    if latest_ckpt and os.path.exists(latest_ckpt):
        print(f"\n📌 Found checkpoint: {os.path.basename(latest_ckpt)}")
        print(f"📌 Resuming from epoch: {latest_epoch}")
        resume_epoch = latest_epoch
        checkpoint_path = latest_ckpt
    else:
        print("\n📭 No checkpoint found. Starting fresh training.")
        checkpoint_path = model_path if os.path.exists(model_path) else None
    
    # Load weights
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"📦 Loading weights: {checkpoint_path}")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(checkpoint_path, map_location=device)
        
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        print(f"✅ Loaded: {len(load_key)} keys")
        print(f"⚠️ Skipped: {len(no_load_key)} keys")
    
    # Loss function
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, anchors_mask)
    
    # Setup logging
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    
    # Mixed precision
    if fp16 and cuda_available:
        from torch.amp import GradScaler
        scaler = GradScaler('cuda')
    else:
        scaler = None
    
    model_train = model.train()
    
    if cuda_available:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
        print("✅ Using GPU")
    
    ema = ModelEMA(model_train)
    
    # Load annotations
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    
    clear_lines = train_lines.copy()
    num_train = len(train_lines)
    num_val = len(val_lines)
    
    print(f"\n📊 Training samples: {num_train}")
    print(f"📊 Validation samples: {num_val}")
    
    # Show config
    show_config(
        classes_path=classes_path, anchors_path=anchors_path,
        anchors_mask=anchors_mask, model_path=checkpoint_path or model_path,
        input_shape=input_shape, Init_Epoch=resume_epoch,
        Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
        Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size,
        Freeze_Train=Freeze_Train, Init_lr=Init_lr, Min_lr=Min_lr,
        optimizer_type=optimizer_type, momentum=momentum,
        lr_decay_type=lr_decay_type, save_period=save_period,
        save_dir=save_dir, num_workers=num_workers,
        num_train=num_train, num_val=num_val
    )
    
    # Training setup
    UnFreeze_flag = False
    Cuda = cuda_available
    
    if Freeze_Train and resume_epoch < Freeze_Epoch:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("❄️ Backbone frozen")
    else:
        UnFreeze_flag = True
        for param in model.backbone.parameters():
            param.requires_grad = True
        print("🔥 Backbone unfrozen")
    
    batch_size = Freeze_batch_size if (Freeze_Train and resume_epoch < Freeze_Epoch) else Unfreeze_batch_size
    
    # Learning rate setup
    nbs = 64
    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
    
    # Optimizer groups
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
    
    if ema:
        ema.updates = epoch_step * resume_epoch
    
    # Dataset
    train_dataset = YoloDataset(
        train_lines, clear_lines, input_shape, num_classes,
        anchors, anchors_mask, epoch_length=UnFreeze_Epoch, train=True
    )
    
    gen = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=yolo_dataset_collate,
        worker_init_fn=partial(worker_init_fn, rank=0, seed=seed)
    )
    
    # Evaluation callback
    eval_callback = EvalCallback(
        model, input_shape, anchors, anchors_mask, class_names,
        num_classes, val_lines, log_dir, Cuda,
        eval_flag=eval_flag, period=eval_period
    )
    
    print("\n" + "=" * 60)
    print("🎯 Starting training loop...")
    print(f"📍 Resume from epoch: {resume_epoch}")
    print(f"📍 Target epoch: {UnFreeze_Epoch}")
    print("=" * 60)
    
    # Training loop
    for epoch in range(resume_epoch, UnFreeze_Epoch):
        
        # Unfreeze backbone after freeze epoch
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
            
            if ema:
                ema.updates = epoch_step * epoch
            
            gen = DataLoader(
                train_dataset, shuffle=True, batch_size=batch_size,
                num_workers=num_workers, pin_memory=True, drop_last=True,
                collate_fn=yolo_dataset_collate,
                worker_init_fn=partial(worker_init_fn, rank=0, seed=seed)
            )
            
            UnFreeze_flag = True
            print("🔥 Backbone unfrozen!")
        
        gen.dataset.epoch_now = epoch
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        
        fit_one_epoch(
            model_train, model, ema, yolo_loss, loss_history,
            eval_callback, optimizer, epoch, epoch_step, gen,
            UnFreeze_Epoch, Cuda, fp16, scaler, save_period,
            save_dir, 0
        )
        
        # Backup checkpoint
        if (epoch + 1) % save_period == 0:
            for f in os.listdir(save_dir):
                if f.startswith(f"ep{epoch + 1:03d}") and f.endswith('.pth'):
                    backup_checkpoint(os.path.join(save_dir, f))
            # Also backup last weights
            last_weights = os.path.join(save_dir, 'last_epoch_weights.pth')
            if os.path.exists(last_weights):
                backup_checkpoint(last_weights)
    
    # Final backup
    print("\n" + "=" * 60)
    print("✅ Training complete!")
    print("💾 Final backup...")
    
    for f in os.listdir(save_dir):
        if f.endswith('.pth'):
            backup_checkpoint(os.path.join(save_dir, f))
    
    print(f"📁 Checkpoints saved to: {CHECKPOINT_DIR}")
    print("📥 Download from Kaggle Output tab")
    print("=" * 60)
    
    loss_history.writer.close()

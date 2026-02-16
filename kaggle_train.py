"""
RDFNet Kaggle Training Script (VOC_FOG_12K_Upload + RTTS eval)
"""
import datetime
import json
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
from utils.dataloader import YoloDataset, yolo_dataset_collate, build_voc_annotation_lines
from utils.utils import (get_anchors, get_classes,
                         seed_everything, show_config, worker_init_fn)
from utils.utils_fit import fit_one_epoch

# ============= KAGGLE CONFIG =============
# Checklist:
# - Generate split: python tools/make_vocfog_split.py --dataset_root /kaggle/input/.../VOC_FOG_12K_Upload
# - Run baseline_pixel: set method_name="baseline_pixel"
# - Run ours_feature: set method_name="ours_feature"
# - Confirm RTTS is test-only: use_rtts_for_training must stay False

# User-specified paths (read-only input)
dataset_root = "/kaggle/input/..."  # user will set
checkpoint_path = "/kaggle/input/datasets/mdhabibourrahman/rdfnet-pth/RDFNet.pth"

# Auto-install required packages for Kaggle
try:
    import subprocess
    import sys
    print("🔧 Checking and installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "thop", "colorama"], 
                         capture_output=True)
    print("✅ Required packages installed!")
except Exception as e:
    print(f"⚠️ Package installation failed: {e}. Continuing with optional imports...")

# Experiment config
training_epochs = 80
method_name = "baseline_pixel"  # or "ours_feature"
use_rtts_for_training = False
alpha_feat = 0.5
lambda_pixel = 0.1
feat_warmup_epochs = 10

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
Freeze_Epoch = 0
Freeze_batch_size = 16
UnFreeze_Epoch = training_epochs
Unfreeze_batch_size = 8
Freeze_Train = False

# Optimizer
Init_lr = 1e-2
Min_lr = Init_lr * 0.01
optimizer_type = "sgd"
momentum = 0.937
weight_decay = 5e-4
lr_decay_type = "cos"

# Save config
save_period = 2  # Save every 2 epochs to prevent loss
save_dir = '/kaggle/working/logs'  # Save outside git repo
eval_flag = True
eval_period = 10
num_workers = 2  # Kaggle has limited workers

def backup_checkpoint(local_path):
    """Backup checkpoint to checkpoint directory"""
    if not os.path.exists(local_path):
        return
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    filename = os.path.basename(local_path)
    shutil.copy2(local_path, os.path.join(CHECKPOINT_DIR, filename))
    print(f"💾 Backed up: {filename}")


def validate_dataset_paths():
    if use_rtts_for_training:
        raise ValueError("use_rtts_for_training must be False: RTTS is test-only.")

    voc_root = os.path.join(dataset_root, "VOC_FOG_12K_Upload")
    rtts_root = os.path.join(dataset_root, "RTTS")

    # Check for splits in multiple possible locations (writable locations first)
    possible_split_locations = [
        os.path.join("/kaggle/working", "ImageSets", "Main"),  # Fixed writable location
        os.path.join("/kaggle/working/dataset_splits", "ImageSets", "Main"),  # From make_vocfog_split.py
        os.path.join(voc_root, "ImageSets", "Main"),  # Original location (read-only)
        os.path.join("/kaggle/working", "VOC_FOG_12K_Upload", "ImageSets", "Main")  # Working directory
    ]
    
    splits_dir = None
    for location in possible_split_locations:
        if os.path.exists(os.path.join(location, "train.txt")):
            splits_dir = location
            print(f"📍 Found dataset splits in: {splits_dir}")
            break
    
    if not splits_dir:
        raise FileNotFoundError("Dataset splits not found. Run make_vocfog_split.py first.")
    
    train_list = os.path.join(splits_dir, "train.txt")
    val_list = os.path.join(splits_dir, "val.txt")
    test_list = os.path.join(splits_dir, "test.txt")
    rtts_test_list = os.path.join(rtts_root, "ImageSets", "Main", "test.txt")

    # Check required paths
    for p in [voc_root, rtts_root]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required path: {p}")
    
    # Check if split files exist
    for split_file in [train_list, val_list, test_list]:
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing split file: {split_file}")

    return voc_root, rtts_root, train_list, val_list, test_list, rtts_test_list


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

    voc_root, rtts_root, train_list_path, val_list_path, test_list_path, rtts_test_list_path = validate_dataset_paths()

    print("\n" + "=" * 60)
    print("🧾 Experiment Summary")
    print(f"Method: {method_name}")
    print(f"Epochs: {training_epochs}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset root: {dataset_root}")
    print(f"Train list: {train_list_path}")
    print(f"Val list: {val_list_path}")
    print(f"VOC_FOG test list: {test_list_path}")
    print(f"RTTS test list: {rtts_test_list_path}")
    print("=" * 60)

    if method_name not in ["baseline_pixel", "ours_feature"]:
        raise ValueError(f"Invalid method_name: {method_name}")

    resume_epoch = 0

    # Load weights (required)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

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
    
    # Load annotations from VOC lists
    train_lines, clean_lines, _ = build_voc_annotation_lines(train_list_path, voc_root, class_names)
    val_lines, _, _ = build_voc_annotation_lines(val_list_path, voc_root, class_names)
    test_lines, _, _ = build_voc_annotation_lines(test_list_path, voc_root, class_names)
    rtts_test_lines, _, _ = build_voc_annotation_lines(rtts_test_list_path, rtts_root, class_names)

    if any("RTTS" in line for line in train_lines) or any("RTTS" in line for line in val_lines):
        raise RuntimeError("RTTS data detected in train/val annotations. RTTS must be test-only.")

    num_train = len(train_lines)
    num_val = len(val_lines)

    print(f"\n📊 Training samples: {num_train}")
    print(f"📊 Validation samples: {num_val}")
    
    # Show config
    show_config(
        classes_path=classes_path, anchors_path=anchors_path,
        anchors_mask=anchors_mask, model_path=checkpoint_path,
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
        train_lines, clean_lines, input_shape, num_classes,
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
            save_dir, 0,
            method_name=method_name,
            alpha_feat=alpha_feat,
            lambda_pixel=lambda_pixel,
            feat_warmup_epochs=feat_warmup_epochs
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

    # Final evaluation on internal VOC_FOG test and external RTTS test
    model_eval = ema.ema if ema else model
    model_eval.eval()

    internal_out = os.path.join(save_dir, "map_internal_vocfog_test")
    external_out = os.path.join(save_dir, "map_external_rtts_test")

    internal_callback = EvalCallback(
        model_eval, input_shape, anchors, anchors_mask, class_names,
        num_classes, test_lines, log_dir, Cuda,
        map_out_path=internal_out, eval_flag=True, period=1, keep_map_out=True
    )
    internal_callback.on_epoch_end(UnFreeze_Epoch, model_eval)
    internal_map = internal_callback.maps[-1]

    external_callback = EvalCallback(
        model_eval, input_shape, anchors, anchors_mask, class_names,
        num_classes, rtts_test_lines, log_dir, Cuda,
        map_out_path=external_out, eval_flag=True, period=1, keep_map_out=True
    )
    external_callback.on_epoch_end(UnFreeze_Epoch, model_eval)
    external_map = external_callback.maps[-1]

    best_val_map = None
    best_epoch = None
    if eval_flag and len(eval_callback.maps) > 1:
        val_maps = eval_callback.maps[1:]
        val_epochs = eval_callback.epoches[1:]
        best_val_map = max(val_maps)
        best_epoch = val_epochs[val_maps.index(best_val_map)]

    summary = {
        "method_name": method_name,
        "epochs": training_epochs,
        "checkpoint": checkpoint_path,
        "alpha_feat": alpha_feat,
        "lambda_pixel": lambda_pixel,
        "best_val_map": best_val_map,
        "best_epoch": best_epoch,
        "internal_vocfog_test_map": internal_map,
        "external_rtts_map": external_map,
        "seed": seed,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    summary_path = os.path.join(save_dir, f"experiment_summary_{time_str}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"🧾 Summary saved: {summary_path}")

    loss_history.writer.close()

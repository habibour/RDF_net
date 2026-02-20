#!/usr/bin/env python3
"""
RDFNet Training on VOC2007 Synthetic Fog Dataset
Fine-tune from checkpoint for 80 epochs on paired fog/clean data.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from functools import partial

from nets.yolo import YoloBody
from nets.yolo_training import (YOLOLoss, get_lr_scheduler, set_optimizer_lr,
                               weights_init)
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import PairedFogCleanDataset, yolo_dataset_collate
from utils.utils import (get_anchors, get_classes, seed_everything,
                        worker_init_fn, get_lr)
from utils.utils_fit import fit_one_epoch

# ============= VOC2007 SYNTHETIC FOG CONFIG =============
# User must set these paths:
dataset_root = ""  # e.g., "/kaggle/input/voc2007-synfog/VOC2007_SYNFOG"
checkpoint_path = ""  # e.g., "/kaggle/input/rdfnet-checkpoint/RDFNet.pth"

# Derived paths
voc2007_clean_root = ""  # Will be set from dataset_root
voc2007_fog_root = ""    # Will be set from dataset_root

# Training configuration
training_epochs = 80
method_name = "ours_feature"  # Feature-level dehazing supervision ONLY
lambda_pixel = 0.1  # Not used in feature method, kept for compatibility
alpha_feat = 0.5    # Feature loss weight
feat_warmup_epochs = 10  # Warmup epochs for feature loss

# Output configuration
save_period = 2
save_dir = '/kaggle/working/logs'
eval_flag = True
eval_period = 5  # Evaluate every 5 epochs

# VOC Classes (20 classes)
VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
               "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

def validate_and_setup_paths():
    """Validate user inputs and setup derived paths."""
    global voc2007_clean_root, voc2007_fog_root
    
    if not dataset_root:
        raise ValueError("❌ dataset_root must be set! Example: '/kaggle/input/voc2007-synfog/VOC2007_SYNFOG'")
    
    if not checkpoint_path:
        raise ValueError("❌ checkpoint_path must be set! Example: '/kaggle/input/rdfnet-checkpoint/RDFNet.pth'")
    
    # Setup derived paths
    voc2007_clean_root = os.path.join(dataset_root, "VOC2007_CLEAN")
    voc2007_fog_root = os.path.join(dataset_root, "VOC2007_FOG")
    
    # Validate paths exist
    required_paths = [
        dataset_root,
        checkpoint_path,
        voc2007_clean_root,
        voc2007_fog_root,
        os.path.join(voc2007_fog_root, "JPEGImages"),
        os.path.join(voc2007_clean_root, "JPEGImages"),
        os.path.join(voc2007_fog_root, "Annotations"),
        os.path.join(voc2007_fog_root, "ImageSets", "Main")
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Required path not found: {path}")
    
    print("✅ All paths validated!")
    print(f"📁 Dataset root: {dataset_root}")
    print(f"🌫️  Fog data: {voc2007_fog_root}")
    print(f"🖼️  Clean data: {voc2007_clean_root}")
    print(f"💾 Checkpoint: {checkpoint_path}")

def setup_train_val_splits():
    """
    Setup train/val splits from VOC2007 ImageSets.
    If only trainval.txt exists, split it 90/10.
    """
    imagesets_dir = os.path.join(voc2007_fog_root, "ImageSets", "Main")
    
    train_list_path = os.path.join(imagesets_dir, "train.txt")
    val_list_path = os.path.join(imagesets_dir, "val.txt")
    trainval_list_path = os.path.join(imagesets_dir, "trainval.txt")
    test_list_path = os.path.join(imagesets_dir, "test.txt")
    
    # Check if train.txt and val.txt exist
    if os.path.exists(train_list_path) and os.path.exists(val_list_path):
        print("✅ Found existing train.txt and val.txt")
    elif os.path.exists(trainval_list_path):
        print("📊 Splitting trainval.txt into train/val (90/10)...")
        
        with open(trainval_list_path, 'r') as f:
            trainval_ids = [line.strip() for line in f.readlines() if line.strip()]
        
        # Deterministic split with seed
        np.random.seed(42)
        np.random.shuffle(trainval_ids)
        
        split_idx = int(0.9 * len(trainval_ids))
        train_ids = trainval_ids[:split_idx]
        val_ids = trainval_ids[split_idx:]
        
        # Write split files
        with open(train_list_path, 'w') as f:
            f.write('\n'.join(train_ids) + '\n')
        
        with open(val_list_path, 'w') as f:
            f.write('\n'.join(val_ids) + '\n')
        
        print(f"   📄 train.txt: {len(train_ids)} samples")
        print(f"   📄 val.txt: {len(val_ids)} samples")
    else:
        raise FileNotFoundError("❌ No train/val splits found. Need train.txt+val.txt or trainval.txt")
    
    # Check test set
    if not os.path.exists(test_list_path):
        print("⚠️  No test.txt found. Will use val set for final evaluation.")
        test_list_path = val_list_path
    
    return train_list_path, val_list_path, test_list_path

def main():
    """Main training function."""
    print("🚀 RDFNet VOC2007 Synthetic Fog Training")
    print("=" * 60)
    
    # Validate setup
    validate_and_setup_paths()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Device: {device}")
    
    # Setup random seeds
    seed_everything(42)
    
    # Setup train/val splits
    train_list_path, val_list_path, test_list_path = setup_train_val_splits()
    
    # Load model configuration
    class_names = VOC_CLASSES
    num_classes = len(class_names)
    anchors_path = 'model_data/yolo_anchors.txt'
    anchors, num_anchors = get_anchors(anchors_path)
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    
    input_shape = [640, 640]
    
    print(f"📊 Classes: {num_classes}")
    print(f"📊 Anchors: {num_anchors}")
    
    # Create model
    model = YoloBody(anchors_mask, num_classes)
    weights_init(model)
    print("🔧 Model initialized")
    
    # Load checkpoint
    if checkpoint_path:
        print(f"📂 Loading checkpoint: {checkpoint_path}")
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
        print(f"✅ Loaded {len(load_key)} parameters from checkpoint")
        if no_load_key:
            print(f"⚠️  Skipped {len(no_load_key)} incompatible parameters")
    
    model = model.to(device)
    
    # Setup loss
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, device, anchors_mask)
    
    # Create datasets
    print(f"\n📚 Creating datasets...")
    
    train_dataset = PairedFogCleanDataset(
        fog_root=voc2007_fog_root,
        clean_root=voc2007_clean_root,
        list_path=train_list_path,
        class_names=class_names,
        input_shape=input_shape,
        num_classes=num_classes,
        anchors=anchors,
        anchors_mask=anchors_mask,
        epoch_length=training_epochs,
        train=True,
        debug=True  # Show debug for first samples
    )
    
    val_dataset = PairedFogCleanDataset(
        fog_root=voc2007_fog_root,
        clean_root=voc2007_clean_root,
        list_path=val_list_path,
        class_names=class_names,
        input_shape=input_shape,
        num_classes=num_classes,
        anchors=anchors,
        anchors_mask=anchors_mask,
        epoch_length=training_epochs,
        train=False
    )
    
    num_train = len(train_dataset)
    num_val = len(val_dataset)
    
    print(f"📊 Training samples: {num_train}")
    print(f"📊 Validation samples: {num_val}")
    
    # Setup training configuration
    batch_size = 4  # Adjust based on GPU memory
    num_workers = 2
    
    # Setup optimizer
    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01
    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 5e-4
    lr_decay_type = "cos"
    
    # Optimizer
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
    
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, training_epochs)
    
    # Setup directories
    os.makedirs(save_dir, exist_ok=True)
    time_str = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    log_dir = os.path.join(save_dir, f"loss_{method_name}_{time_str}")
    
    # Setup callbacks
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    
    eval_callback = EvalCallback(
        model, input_shape, anchors, anchors_mask, class_names,
        num_classes, val_dataset.fog_lines, log_dir, device,
        eval_flag=eval_flag, period=eval_period
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        collate_fn=yolo_dataset_collate,
        worker_init_fn=partial(worker_init_fn, rank=0, seed=42)
    )
    
    val_loader = DataLoader(
        val_dataset, shuffle=False, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        collate_fn=yolo_dataset_collate
    )
    
    epoch_step = len(train_loader)
    epoch_step_val = len(val_loader)
    
    # Training setup
    Cuda = True
    fp16 = True
    scaler = torch.cuda.amp.GradScaler() if fp16 else None
    
    model_train = model.train()
    
    print(f"\n🎯 Training Configuration:")
    print(f"   Method: {method_name} (Feature-level dehazing supervision)")
    print(f"   Epochs: {training_epochs}")
    print(f"   Feature loss weight (α): {alpha_feat}")
    print(f"   Feature warmup epochs: {feat_warmup_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {Init_lr_fit} -> {Min_lr_fit}")
    print(f"   Save every: {save_period} epochs")
    print(f"   Evaluate every: {eval_period} epochs")
    
    # Training loop
    print(f"\n🚀 Starting training...")
    print("=" * 60)
    
    for epoch in range(training_epochs):
        train_dataset.epoch_now = epoch
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        
        fit_one_epoch(
            model_train, model, None, yolo_loss, loss_history,
            eval_callback, optimizer, epoch, epoch_step, train_loader,
            training_epochs, Cuda, fp16, scaler, save_period,
            save_dir, 0,
            method_name=method_name,
            alpha_feat=alpha_feat,
            lambda_pixel=lambda_pixel,
            feat_warmup_epochs=feat_warmup_epochs
        )
        
        # Save checkpoint
        if (epoch + 1) % save_period == 0 or epoch == training_epochs - 1:
            print(f"💾 Saving checkpoint at epoch {epoch + 1}")
            checkpoint_name = f"ep{epoch + 1:03d}-{method_name}.pth"
            torch.save(model.state_dict(), os.path.join(save_dir, checkpoint_name))
    
    # Final evaluation and summary
    print(f"\n✅ Training complete!")
    
    # Load best model if available
    best_model_path = os.path.join(log_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"📂 Loading best model for final evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    
    # Final evaluation on test set
    test_dataset = PairedFogCleanDataset(
        fog_root=voc2007_fog_root,
        clean_root=voc2007_clean_root,
        list_path=test_list_path,
        class_names=class_names,
        input_shape=input_shape,
        num_classes=num_classes,
        anchors=anchors,
        anchors_mask=anchors_mask,
        epoch_length=1,
        train=False
    )
    
    final_eval_callback = EvalCallback(
        model, input_shape, anchors, anchors_mask, class_names,
        num_classes, test_dataset.fog_lines, save_dir, device,
        eval_flag=True, period=1
    )
    
    print(f"🔍 Final evaluation on test set ({len(test_dataset)} samples)...")
    final_eval_callback.on_epoch_end(0, model)
    
    # Write experiment summary
    summary = {
        'experiment': 'RDFNet_VOC2007_SynFog_FeatureSupervision',
        'method_name': method_name,
        'description': 'Feature-level dehazing supervision using detector backbone features',
        'epochs': training_epochs,
        'checkpoint': checkpoint_path,
        'dataset_root': dataset_root,
        'train_samples': num_train,
        'val_samples': num_val,
        'test_samples': len(test_dataset),
        'loss_formulation': f'L_total = L_detection + α_feat * warmup * Σ L1(F_l(restored), F_l(clean))',
        'parameters': {
            'alpha_feat': alpha_feat,
            'feat_warmup_epochs': feat_warmup_epochs,
            'warmup_schedule': 'linear from 0 to 1 over warmup epochs'
        },
        'training_config': {
            'batch_size': batch_size,
            'learning_rate': f"{Init_lr_fit} -> {Min_lr_fit}",
            'optimizer': optimizer_type,
            'lr_decay': lr_decay_type
        }
    }
    
    # Add best validation mAP if available
    best_log_path = os.path.join(log_dir, "best_model_log.txt")
    if os.path.exists(best_log_path):
        with open(best_log_path, 'r') as f:
            content = f.read()
            if 'mAP:' in content:
                map_line = [line for line in content.split('\n') if 'mAP:' in line][0]
                summary['best_val_mAP'] = float(map_line.split('mAP:')[1].strip())
    
    # Save experiment summary
    summary_path = os.path.join(save_dir, f"experiment_summary_feature_supervision.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Experiment summary saved: {summary_path}")
    print(f"🎯 Feature-level supervision training completed!")
    print("=" * 60)

if __name__ == '__main__':
    main()
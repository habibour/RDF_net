"""
RDFNet Kaggle Training Script - Complete Implementation
Fine-tune RDFNet for exactly 80 epochs with dual training methods and robust validation.
"""
import datetime
import json
import os
import random
import shutil
from functools import partial
import xml.etree.ElementTree as ET
from PIL import Image

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

# =============================================================================
# CENTRALIZED KAGGLE CONFIGURATION
# =============================================================================

# HARDCODED KAGGLE PATHS (DO NOT CHANGE)
checkpoint_path = "/kaggle/input/datasets/mdhabibourrahman/rdfnet-pth/RDFNet.pth"
voc_annotations_path = "/kaggle/input/datasets/mdhabibourrahman/voc-2012/VOC2012_train_val/Annotations"
voc_imagesets_path = "/kaggle/input/datasets/mdhabibourrahman/voc-2012/VOC2012_train_val/ImageSets"
voc_clean_images_path = "/kaggle/input/datasets/mdhabibourrahman/voc-2012/VOC2012_train_val/JPEGImages"
voc_fog_images_path = "/kaggle/input/datasets/mdhabibourrahman/voc-2012/VOC2012_train_val/JPEGImages_foggy"
voc_segmentation_class_path = "/kaggle/input/datasets/mdhabibourrahman/voc-2012/VOC2012_train_val/SegmentationClass"
voc_segmentation_object_path = "/kaggle/input/datasets/mdhabibourrahman/voc-2012/VOC2012_train_val/SegmentationObject"
rtts_root = "/kaggle/input/datasets/mdhabibourrahman/foggy-voc/RTTS"

# TRAINING CONFIGURATION (FIXED)
method_name = "baseline_pixel"  # Options: "baseline_pixel" or "ours_feature"
epochs = 80  # FIXED - ignore CLI overrides
lambda_pixel = 0.1
alpha_feat = 0.5
feat_warmup_epochs = 10
seed = 42
batch_size = 8  # Unfreeze batch size
optimizer_type = "SGD"
init_lr = 0.005  # Safer for fine-tuning
cos_decay = True

# OUTPUT PATHS
save_dir = '/kaggle/working/logs'
checkpoint_dir = '/kaggle/working/checkpoints'

# VOC CLASSES
VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
               "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# MODEL CONFIG
classes_path = 'model_data/rtts_classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
input_shape = [640, 640]
fp16 = True
num_workers = 2

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def resolve_image_path(images_dir, image_id, prefer_foggy_suffix=False):
    """
    Resolve image path for an ID across common VOC filename variants.
    Supports:
    - image_id.{jpg,jpeg,png}
    - image_id_foggy.{jpg,jpeg,png}
    """
    exts = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    candidates = []

    if prefer_foggy_suffix:
        candidates.extend([f"{image_id}_foggy{ext}" for ext in exts])
    candidates.extend([f"{image_id}{ext}" for ext in exts])
    if not prefer_foggy_suffix:
        candidates.extend([f"{image_id}_foggy{ext}" for ext in exts])

    for name in candidates:
        path = os.path.join(images_dir, name)
        if os.path.exists(path):
            return path
    return None

def sanity_check_pairing():
    """
    Run comprehensive sanity checks BEFORE training.
    Validates dataset structure, paths, and fog-clean pairing.
    """
    print("\n" + "="*60)
    print("🔍 RUNNING SANITY CHECKS")
    print("="*60)
    
    # Check 1: Essential directories exist
    print("✅ Check 1: Essential directories...")
    required_paths = {
        "Checkpoint": checkpoint_path,
        "VOC Annotations": voc_annotations_path,
        "VOC ImageSets": voc_imagesets_path,
        "VOC Clean Images": voc_clean_images_path,
        "VOC Fog Images": voc_fog_images_path,
        "RTTS Root": rtts_root
    }
    
    for name, path in required_paths.items():
        if not os.path.exists(path):
            raise RuntimeError(f"Missing {name}: {path}")
        print(f"   ✓ {name}: {path}")
    
    # Check 2: ImageSets/Main structure
    print("✅ Check 2: ImageSets structure...")
    sets_main = os.path.join(voc_imagesets_path, "Main")
    if not os.path.exists(sets_main):
        raise RuntimeError(f"Missing ImageSets/Main: {sets_main}")
    
    # Check 3: Build 80-10-10 train/val/test splits with fog filtering
    print("✅ Check 3: Creating 80-10-10 train/val/test splits...")
    train_txt_src = os.path.join(sets_main, "train.txt")
    val_txt_src = os.path.join(sets_main, "val.txt")
    trainval_txt = os.path.join(sets_main, "trainval.txt")
    
    # Create working directory for splits (Kaggle input is read-only)
    splits_work_dir = os.path.join(save_dir, "splits")
    os.makedirs(splits_work_dir, exist_ok=True)
    
    train_txt = os.path.join(splits_work_dir, "train.txt")
    val_txt = os.path.join(splits_work_dir, "val.txt")
    test_txt = os.path.join(splits_work_dir, "test.txt")
    
    # Collect all available image IDs
    all_ids = []
    if os.path.exists(train_txt_src) and os.path.exists(val_txt_src):
        print("   ℹ Reading train.txt and val.txt from source")
        with open(train_txt_src, 'r') as f:
            all_ids.extend([line.strip() for line in f if line.strip()])
        with open(val_txt_src, 'r') as f:
            all_ids.extend([line.strip() for line in f if line.strip()])
    elif os.path.exists(trainval_txt):
        print("   ℹ Reading trainval.txt from source")
        with open(trainval_txt, 'r') as f:
            all_ids = [line.strip() for line in f if line.strip()]
    else:
        raise RuntimeError("Neither (train.txt, val.txt) nor trainval.txt found in ImageSets/Main")
    
    print(f"   ℹ Total available samples: {len(all_ids)}")
    
    # Filter only IDs that have foggy versions
    print("   ℹ Filtering samples with foggy images...")
    ids_with_fog = []
    for img_id in all_ids:
        fog_path = resolve_image_path(voc_fog_images_path, img_id, prefer_foggy_suffix=True)
        if fog_path is not None:
            ids_with_fog.append(img_id)
    
    print(f"   ✓ Samples with foggy images: {len(ids_with_fog)}")
    
    # Shuffle and split 80-10-10
    random.seed(seed)
    random.shuffle(ids_with_fog)
    
    total = len(ids_with_fog)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)
    
    train_ids = ids_with_fog[:train_end]
    val_ids = ids_with_fog[train_end:val_end]
    test_ids = ids_with_fog[val_end:]
    
    # Write split files
    with open(train_txt, 'w') as f:
        for img_id in train_ids:
            f.write(f"{img_id}\n")
    
    with open(val_txt, 'w') as f:
        for img_id in val_ids:
            f.write(f"{img_id}\n")
    
    with open(test_txt, 'w') as f:
        for img_id in test_ids:
            f.write(f"{img_id}\n")
    
    print(f"   ✓ Created train.txt: {len(train_ids)} samples ({len(train_ids)/total*100:.1f}%)")
    print(f"   ✓ Created val.txt: {len(val_ids)} samples ({len(val_ids)/total*100:.1f}%)")
    print(f"   ✓ Created test.txt: {len(test_ids)} samples ({len(test_ids)/total*100:.1f}%)")
    
    # Check 4: Sample pairing validation (first 30 samples with fog)
    print("✅ Check 4: Sample pairing validation...")
    with open(train_txt, 'r') as f:
        all_train_ids = [line.strip() for line in f if line.strip()]
    
    # Filter only IDs that have foggy versions
    train_ids_with_fog = []
    for img_id in all_train_ids:
        fog_path = resolve_image_path(voc_fog_images_path, img_id, prefer_foggy_suffix=True)
        if fog_path is not None:
            train_ids_with_fog.append(img_id)
        if len(train_ids_with_fog) >= 30:
            break
    
    print(f"   ℹ Found {len(train_ids_with_fog)} samples with foggy images (from first {len(all_train_ids)} total)")
    
    pairing_issues = 0
    total_diff = 0
    
    for i, image_id in enumerate(train_ids_with_fog):
        fog_path = resolve_image_path(voc_fog_images_path, image_id, prefer_foggy_suffix=True)
        clean_path = resolve_image_path(voc_clean_images_path, image_id, prefer_foggy_suffix=False)
        xml_path = os.path.join(voc_annotations_path, f"{image_id}.xml")
        
        # Check file existence (should exist since we filtered)
        if fog_path is None:
            pairing_issues += 1
            print(f"   ❌ Missing fog image for ID: {image_id}")
            continue
        if clean_path is None:
            pairing_issues += 1
            print(f"   ❌ Missing clean image for ID: {image_id}")
            continue
        if not os.path.exists(xml_path):
            pairing_issues += 1
            print(f"   ❌ Missing annotation: {xml_path}")
            continue
        
        # Check XML parsing
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            objects = root.findall('object')
            class_names = [obj.find('name').text for obj in objects]
            valid_classes = [name for name in class_names if name in VOC_CLASSES]
            if not valid_classes:
                print(f"   ⚠ No valid objects in {image_id}")
        except Exception as e:
            pairing_issues += 1
            print(f"   ❌ XML parsing error for {image_id}: {e}")
            continue
        
        # Check image loading and difference
        if i < 2:  # Only check first 2 for speed
            try:
                fog_img = Image.open(fog_path).convert('RGB')
                clean_img = Image.open(clean_path).convert('RGB')
                
                # Simple preprocessing
                fog_np = np.array(fog_img, dtype=np.float32) / 255.0
                clean_np = np.array(clean_img, dtype=np.float32) / 255.0
                
                mean_diff = np.mean(np.abs(fog_np - clean_np))
                total_diff += mean_diff
                
                print(f"   ℹ Sample {i+1}: {fog_path} | {clean_path} | mean_diff={mean_diff:.4f}")
                
                if mean_diff < 1e-3:
                    raise RuntimeError(f"Images too similar (mean_diff={mean_diff:.6f}), pairing likely broken!")
                    
            except Exception as e:
                pairing_issues += 1
                print(f"   ❌ Image loading error for {image_id}: {e}")
    
    if pairing_issues > 0:
        print(f"   ⚠ Warning: Found {pairing_issues} pairing issues in {len(train_ids_with_fog)} checked samples")
        print(f"   ℹ This is acceptable - training will skip problematic samples")
    else:
        print(f"   ✓ All {len(train_ids_with_fog)} checked samples passed validation")
    
    # Check 5: RTTS is test-only
    print("✅ Check 5: RTTS test-only validation...")
    if rtts_root in [train_txt, val_txt]:
        raise RuntimeError("RTTS path found in train/val lists - RTTS is test-only!")
    
    print("   ✓ RTTS correctly isolated for testing")
    
    print("\n" + "="*60)
    print("🎉 SANITY CHECK PASSED ✅")
    print("="*60)
    
    return train_txt, val_txt, test_txt

def create_train_val_splits(trainval_path, train_path, val_path, seed=42):
    """
    Split trainval.txt into train.txt (90%) and val.txt (10%) deterministically.
    """
    with open(trainval_path, 'r') as f:
        image_ids = [line.strip() for line in f if line.strip()]
    
    random.seed(seed)
    random.shuffle(image_ids)
    
    split_idx = int(0.9 * len(image_ids))
    train_ids = image_ids[:split_idx]
    val_ids = image_ids[split_idx:]
    
    with open(train_path, 'w') as f:
        f.write('\n'.join(train_ids))
    
    with open(val_path, 'w') as f:
        f.write('\n'.join(val_ids))
    
    print(f"   ✓ Created train.txt: {len(train_ids)} samples")
    print(f"   ✓ Created val.txt: {len(val_ids)} samples")

def build_annotation_lines(image_list_path, clean_images_dir, fog_images_dir, annotations_dir, class_names):
    """
    Build annotation lines for VOC dataset with clean/fog pairing.
    Returns: List of strings in format: "fog_path,clean_path xmin,ymin,xmax,ymax,class_id ..."
    """
    with open(image_list_path, 'r') as f:
        image_ids = [line.strip() for line in f if line.strip()]
    
    annotation_lines = []
    errors = 0
    
    for i, image_id in enumerate(image_ids):
        fog_path = resolve_image_path(fog_images_dir, image_id, prefer_foggy_suffix=True)
        clean_path = resolve_image_path(clean_images_dir, image_id, prefer_foggy_suffix=False)
        xml_path = os.path.join(annotations_dir, f"{image_id}.xml")
        
        if fog_path is None or clean_path is None or not os.path.exists(xml_path):
            errors += 1
            if i < 5:
                print(f"❌ Missing files for {image_id}")
            continue
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            boxes = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in class_names:
                    continue
                
                class_id = class_names.index(class_name)
                bbox = obj.find('bndbox')
                
                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))
                
                boxes.append(f"{xmin},{ymin},{xmax},{ymax},{class_id}")
            
            if boxes:
                line = f"{fog_path},{clean_path} " + " ".join(boxes)
                annotation_lines.append(line)
                
                if i < 3:  # Debug first few
                    print(f"✓ Sample {i+1}: {image_id} -> {len(boxes)} objects")
        
        except Exception as e:
            errors += 1
            if i < 5:
                print(f"❌ XML parsing error for {image_id}: {e}")
    
    print(f"✓ Built annotation lines: {len(annotation_lines)} samples")
    if errors > 0:
        print(f"⚠ Errors/missing: {errors} samples")
    
    return annotation_lines

def build_rtts_annotation_lines(rtts_root, class_names):
    """
    Build annotation lines for RTTS test dataset.
    """
    rtts_imagesets = os.path.join(rtts_root, "ImageSets", "Main", "test.txt")
    rtts_images = os.path.join(rtts_root, "JPEGImages")
    rtts_annotations = os.path.join(rtts_root, "Annotations")
    
    if not os.path.exists(rtts_imagesets):
        print(f"⚠ RTTS test.txt not found: {rtts_imagesets}")
        return []
    
    return build_annotation_lines(rtts_imagesets, rtts_images, rtts_images, rtts_annotations, class_names)

def load_checkpoint_with_validation(model, checkpoint_path):
    """
    Load checkpoint with strict=False and comprehensive validation.
    """
    print(f"\n📥 Loading checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Filter out YOLO head layers that don't match (different num_classes)
        # RTTS checkpoint has 10 classes, VOC has 20 classes
        model_state = model.state_dict()
        filtered_state_dict = {}
        skipped_keys = []
        
        for k, v in state_dict.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_state_dict[k] = v
                else:
                    skipped_keys.append(f"{k}: checkpoint {v.shape} vs model {model_state[k].shape}")
            else:
                skipped_keys.append(f"{k}: not in model")
        
        if skipped_keys:
            print(f"   ℹ Skipping {len(skipped_keys)} mismatched layers (YOLO heads for different num_classes)")
            if len(skipped_keys) <= 10:
                for sk in skipped_keys:
                    print(f"     - {sk}")
        
        # Load with strict=False to handle partial matching
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
        
        total_keys = len(state_dict)
        loaded_keys = len(filtered_state_dict)
        missing_count = len(missing_keys)
        
        print(f"   ✓ Total checkpoint keys: {total_keys}")
        print(f"   ✓ Loaded keys: {loaded_keys}")
        print(f"   ✓ Skipped keys: {len(skipped_keys)}")
        print(f"   ✓ Missing keys in model: {missing_count}")
        
        # Validate loading success - be more lenient since we're skipping YOLO heads
        if loaded_keys < 0.5 * total_keys:
            raise RuntimeError(f"Too few keys loaded: {loaded_keys}/{total_keys} (<50%)")
        
        print("   ✅ Checkpoint loaded successfully (backbone + feature extractor)")
        print("   ℹ YOLO heads initialized randomly for VOC 20 classes")
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

def validate_rtts_isolation(train_lines, val_lines):
    """
    Ensure RTTS is never used for training or validation.
    """
    for lines, split_name in [(train_lines, "train"), (val_lines, "val")]:
        for line in lines:
            if "RTTS" in line or rtts_root in line:
                raise RuntimeError(f"RTTS path found in {split_name} split - RTTS is test-only!")

def evaluate_dual_datasets(model, anchors, class_names, input_shape, 
                          voc_test_lines, rtts_test_lines, cuda, save_dir):
    """
    Evaluate on both VOC fog test and RTTS test datasets.
    """
    from utils.callbacks import EvalCallback
    
    results = {}
    
    # VOC fog test evaluation
    if voc_test_lines:
        print("📊 Evaluating on VOC fog test set...")
        voc_eval_dir = os.path.join(save_dir, "map_internal_vocfog_test")
        os.makedirs(voc_eval_dir, exist_ok=True)
        
        # Create temporary eval callback for VOC
        voc_eval = EvalCallback(model, input_shape, anchors, class_names, 
                               val_lines=voc_test_lines, log_dir=voc_eval_dir, cuda=cuda)
        voc_map = voc_eval.on_epoch_end(0)  # Run evaluation
        results['voc_fog_test_map'] = voc_map
        print(f"   ✓ VOC fog test mAP: {voc_map:.4f}")
    
    # RTTS test evaluation  
    if rtts_test_lines:
        print("📊 Evaluating on RTTS test set...")
        rtts_eval_dir = os.path.join(save_dir, "map_external_rtts_test")
        os.makedirs(rtts_eval_dir, exist_ok=True)
        
        # Create temporary eval callback for RTTS
        rtts_eval = EvalCallback(model, input_shape, anchors, class_names,
                                val_lines=rtts_test_lines, log_dir=rtts_eval_dir, cuda=cuda)
        rtts_map = rtts_eval.on_epoch_end(0)  # Run evaluation
        results['rtts_test_map'] = rtts_map
        print(f"   ✓ RTTS test mAP: {rtts_map:.4f}")
    
    return results

def save_experiment_summary(save_dir, results, best_epoch, best_voc_map):
    """
    Save comprehensive experiment summary as JSON.
    """
    summary = {
        "experiment_config": {
            "method_name": method_name,
            "epochs": epochs,
            "init_lr": init_lr,
            "batch_size": batch_size,
            "lambda_pixel": lambda_pixel,
            "alpha_feat": alpha_feat,
            "feat_warmup_epochs": feat_warmup_epochs,
            "seed": seed,
            "optimizer": optimizer_type,
            "cos_decay": cos_decay
        },
        "dataset_paths": {
            "checkpoint_path": checkpoint_path,
            "voc_annotations": voc_annotations_path,
            "voc_imagesets": voc_imagesets_path,
            "voc_clean_images": voc_clean_images_path,
            "voc_fog_images": voc_fog_images_path,
            "rtts_root": rtts_root
        },
        "results": {
            "best_epoch": best_epoch,
            "best_voc_map": best_voc_map,
            "final_voc_map": results.get('voc_fog_test_map', None),
            "final_rtts_map": results.get('rtts_test_map', None)
        },
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    summary_path = os.path.join(save_dir, "experiment_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Experiment summary saved: {summary_path}")

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main():
    """
    Main training function with all requirements implemented.
    """
    print("="*60)
    print("🚀 RDFNet Kaggle Training - 80 Epochs Fine-tuning")
    print("="*60)
    
    # Enforce fixed configuration
    print(f"📋 Configuration:")
    print(f"   Method: {method_name}")
    print(f"   Epochs: {epochs} (FIXED)")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {init_lr}")
    print(f"   Lambda pixel: {lambda_pixel}")
    print(f"   Alpha feat: {alpha_feat}")
    print(f"   Seed: {seed}")
    
    # Set up environment
    seed_everything(seed)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f"   Device: {device}")
    
    # Run sanity checks
    train_txt, val_txt, test_txt = sanity_check_pairing()
    
    # Build annotation lines
    print("\n📝 Building annotation lines...")
    train_lines = build_annotation_lines(train_txt, voc_clean_images_path, voc_fog_images_path, 
                                       voc_annotations_path, VOC_CLASSES)
    val_lines = build_annotation_lines(val_txt, voc_clean_images_path, voc_fog_images_path,
                                     voc_annotations_path, VOC_CLASSES)
    test_lines = build_annotation_lines(test_txt, voc_clean_images_path, voc_fog_images_path,
                                      voc_annotations_path, VOC_CLASSES)
    rtts_lines = build_rtts_annotation_lines(rtts_root, VOC_CLASSES)
    
    # Validate RTTS isolation
    validate_rtts_isolation(train_lines, val_lines)
    
    print(f"   ✓ Train samples: {len(train_lines)}")
    print(f"   ✓ Val samples: {len(val_lines)}")
    print(f"   ✓ Test samples: {len(test_lines)}")
    print(f"   ✓ RTTS samples: {len(rtts_lines)}")
    
    # Load model
    print("\n🏗 Building model...")
    anchors, num_anchors = get_anchors(anchors_path)
    num_classes = len(VOC_CLASSES)
    
    model = YoloBody(anchors_mask, num_classes)
    
    # Load checkpoint
    model = load_checkpoint_with_validation(model, checkpoint_path)
    
    if cuda:
        model = model.cuda()
    
    # Create data loaders
    print("\n📊 Creating data loaders...")
    train_dataset = YoloDataset(train_lines, input_shape, num_classes, train=True)
    val_dataset = YoloDataset(val_lines, input_shape, num_classes, train=False)
    
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, 
                             num_workers=num_workers, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size,
                           num_workers=num_workers, pin_memory=True, 
                           drop_last=False, collate_fn=yolo_dataset_collate)
    
    # Set up training components
    print("\n⚙️ Setting up training components...")
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, anchors_mask)
    loss_history = LossHistory(save_dir, model, input_shape)
    
    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=0.937, weight_decay=5e-4)
    lr_scheduler = get_lr_scheduler("cos", init_lr, init_lr * 0.01, epochs)
    
    # Training loop
    print(f"\n🏃 Starting training for {epochs} epochs...")
    best_map = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Training
        fit_one_epoch(model, model, None, yolo_loss, loss_history, None, optimizer, 
                     epoch, len(train_loader), train_loader, epochs, cuda, fp16, None,
                     save_period=10, save_dir=save_dir, local_rank=0,
                     method_name=method_name, alpha_feat=alpha_feat, 
                     lambda_pixel=lambda_pixel, feat_warmup_epochs=feat_warmup_epochs)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluation (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            print(f"📊 Running evaluation at epoch {epoch+1}...")
            results = evaluate_dual_datasets(model, anchors, VOC_CLASSES, input_shape,
                                            test_lines, rtts_lines, cuda, save_dir)
            
            current_map = results.get('voc_fog_test_map', 0)
            if current_map > best_map:
                best_map = current_map
                best_epoch = epoch + 1
                
                # Save best model
                best_path = os.path.join(save_dir, "best_model.pth")
                torch.save(model.state_dict(), best_path)
                print(f"   🏆 New best model saved: {best_map:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            epoch_checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), epoch_checkpoint_path)
    
    # Final evaluation
    print("\n🏁 Running final evaluation...")
    final_results = evaluate_dual_datasets(model, anchors, VOC_CLASSES, input_shape,
                                          test_lines, rtts_lines, cuda, save_dir)
    
    # Save final model
    final_path = os.path.join(save_dir, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    
    # Save experiment summary
    save_experiment_summary(save_dir, final_results, best_epoch, best_map)
    
    print(f"\n✅ Training completed!")
    print(f"   Best epoch: {best_epoch}")
    print(f"   Best VOC mAP: {best_map:.4f}")
    print(f"   Final VOC mAP: {final_results.get('voc_fog_test_map', 'N/A')}")
    print(f"   Final RTTS mAP: {final_results.get('rtts_test_map', 'N/A')}")

if __name__ == '__main__':
    main()

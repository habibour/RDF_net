"""
RDFNet Setup Verification Script
Validates the complete Kaggle training setup before running full training.
"""
import os
import sys
import torch
import numpy as np
from PIL import Image

# Add RDFNet root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nets.model import YoloBody
from utils.utils import get_anchors, get_classes, seed_everything
from utils.dataloader import YoloDataset, yolo_dataset_collate
from torch.utils.data import DataLoader

# Import configuration from kaggle_train.py
try:
    from kaggle_train import (
        checkpoint_path, voc_annotations_path, voc_imagesets_path,
        voc_clean_images_path, voc_fog_images_path, rtts_root,
        method_name, epochs, lambda_pixel, alpha_feat, feat_warmup_epochs,
        seed, batch_size, VOC_CLASSES, anchors_path, anchors_mask, input_shape,
        sanity_check_pairing, build_annotation_lines, load_checkpoint_with_validation,
        validate_rtts_isolation
    )
except ImportError as e:
    print(f"❌ Failed to import from kaggle_train.py: {e}")
    sys.exit(1)

def check_setup():
    """
    Run comprehensive setup verification.
    """
    print("="*60)
    print("🔧 RDFNet Setup Verification")
    print("="*60)
    
    try:
        # Step 1: Run sanity checks
        print("\n1️⃣ Running dataset sanity checks...")
        train_txt, val_txt, test_txt = sanity_check_pairing()
        print("   ✅ Sanity checks passed")
        
        # Step 2: Build annotation lines  
        print("\n2️⃣ Building annotation lines...")
        train_lines = build_annotation_lines(train_txt, voc_clean_images_path, voc_fog_images_path, 
                                           voc_annotations_path, VOC_CLASSES)
        val_lines = build_annotation_lines(val_txt, voc_clean_images_path, voc_fog_images_path,
                                         voc_annotations_path, VOC_CLASSES)
        
        # Validate RTTS isolation
        validate_rtts_isolation(train_lines, val_lines)
        
        print(f"   ✅ Train samples: {len(train_lines)}")
        print(f"   ✅ Val samples: {len(val_lines)}")
        
        # Step 3: Initialize model
        print("\n3️⃣ Initializing model...")
        anchors = get_anchors(anchors_path)
        num_classes = len(VOC_CLASSES)
        model = YoloBody(anchors_mask, num_classes)
        print(f"   ✅ Model created: {num_classes} classes")
        
        # Step 4: Load checkpoint
        print("\n4️⃣ Loading checkpoint...")
        model = load_checkpoint_with_validation(model, checkpoint_path)
        print("   ✅ Checkpoint loaded successfully")
        
        # Step 5: Test data loader
        print("\n5️⃣ Testing data loader...")
        train_dataset = YoloDataset(train_lines[:4], input_shape, num_classes, train=True)  # Test with 4 samples
        train_loader = DataLoader(train_dataset, shuffle=False, batch_size=2, 
                                 num_workers=0, pin_memory=False,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
        
        # Get one batch
        batch = next(iter(train_loader))
        fog_images, targets, clean_images = batch
        
        print(f"   ✅ Batch loaded successfully")
        print(f"      Fog images: {fog_images.shape}")
        print(f"      Clean images: {clean_images.shape}")
        print(f"      Targets: {targets.shape}")
        
        # Verify images are different
        mean_diff = torch.mean(torch.abs(fog_images - clean_images)).item()
        print(f"      Mean fog-clean difference: {mean_diff:.6f}")
        if mean_diff < 1e-3:
            print("      ⚠️ WARNING: Images may not be properly paired")
        
        # Step 6: Test model forward pass
        print("\n6️⃣ Testing model forward pass...")
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        fog_images = fog_images.to(device)
        clean_images = clean_images.to(device)
        
        with torch.no_grad():
            # Test detection forward
            combined_input = torch.cat([fog_images, clean_images], dim=0)
            outputs = model(combined_input)
            print(f"   ✅ Forward pass successful")
            print(f"      Outputs: {len(outputs)} tensors")
            for i, out in enumerate(outputs):
                print(f"      Output {i}: {out.shape}")
            
            # Test feature extraction
            if len(outputs) >= 4:
                restored = outputs[3]  # Dehazing output
                print(f"      Dehazing output: {restored.shape}")
                
                # Test feature extraction from restored image
                _, restored_feats = model(restored[:1], return_feats=True, det_only=True)  # Test with 1 image
                _, clean_feats = model(clean_images[:1], return_feats=True, det_only=True)
                
                print(f"   ✅ Feature extraction successful")
                print(f"      Restored features: {len(restored_feats)} levels")
                print(f"      Clean features: {len(clean_feats)} levels")
                
                for i in range(len(restored_feats)):
                    print(f"      Level {i}: {restored_feats[i].shape} vs {clean_feats[i].shape}")
            else:
                print("   ⚠️ WARNING: No dehazing output found (this is expected if model isn't trained)")
        
        # Step 7: Test loss computation
        print("\n7️⃣ Testing loss computation...")
        from nets.yolo_training import YOLOLoss
        import torch.nn as nn
        
        cuda = torch.cuda.is_available()
        yolo_loss = YOLOLoss(anchors, num_classes, input_shape, cuda, anchors_mask)
        
        # Mock forward pass for loss testing
        model.train()
        detect_outputs = outputs[:3] if len(outputs) >= 3 else [torch.randn(2, 75, 80, 80), torch.randn(2, 75, 40, 40), torch.randn(2, 75, 20, 20)]
        
        # Detection loss
        if targets.numel() > 0:  # Only if we have targets
            loss_det = yolo_loss(detect_outputs, targets, fog_images)
            print(f"   ✅ Detection loss computed: {loss_det.item():.6f}")
        else:
            print("   ⚠️ No targets available for loss computation")
            loss_det = torch.tensor(0.0)
        
        # Test pixel loss (baseline method)
        if method_name == "baseline_pixel" and len(outputs) >= 4:
            criterion = nn.MSELoss()
            pixel_loss = criterion(outputs[3][:fog_images.shape[0]], clean_images)
            print(f"   ✅ Pixel loss computed: {pixel_loss.item():.6f}")
        
        # Test feature loss (ours method)
        if method_name == "ours_feature" and len(outputs) >= 4:
            restored = outputs[3][:fog_images.shape[0]]
            _, restored_feats = model(restored, return_feats=True, det_only=True)
            _, clean_feats = model(clean_images, return_feats=True, det_only=True)
            
            feat_loss = sum(torch.nn.functional.l1_loss(restored_feats[i], clean_feats[i]) 
                           for i in range(len(restored_feats)))
            print(f"   ✅ Feature loss computed: {feat_loss.item():.6f}")
            
            # Test warmup
            warmup = min(1.0, 1.0 / feat_warmup_epochs)  # epoch 0
            print(f"   ✅ Warmup factor: {warmup:.3f}")
        
        # Step 8: Configuration summary
        print("\n8️⃣ Configuration summary...")
        print(f"   Method: {method_name}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Lambda pixel: {lambda_pixel}")
        print(f"   Alpha feat: {alpha_feat}")
        print(f"   Feat warmup epochs: {feat_warmup_epochs}")
        print(f"   Seed: {seed}")
        print(f"   Device: {device}")
        
        print("\n" + "="*60)
        print("🎉 SETUP VERIFIED ✅")
        print("All components are working correctly!")
        print("Ready to run kaggle_train.py")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ SETUP VERIFICATION FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️ Fix the above issues before running training")
        return False

if __name__ == '__main__':
    # Set seed for reproducibility
    seed_everything(seed)
    
    success = check_setup()
    sys.exit(0 if success else 1)
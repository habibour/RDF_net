#!/usr/bin/env python3
"""
Quick setup script for VOC2007 synthetic fog training.
Validates setup and provides guidance.
"""

import os
import sys
import argparse

def check_voc2007_structure(voc_root):
    """Check if VOC2007 has required structure."""
    required = ['JPEGImages', 'Annotations', 'ImageSets']
    missing = []
    
    for req in required:
        if not os.path.exists(os.path.join(voc_root, req)):
            missing.append(req)
    
    return missing

def check_generated_dataset(synfog_root):
    """Check if synthetic fog dataset exists and is valid."""
    clean_root = os.path.join(synfog_root, 'VOC2007_CLEAN')
    fog_root = os.path.join(synfog_root, 'VOC2007_FOG')
    
    if not os.path.exists(clean_root) or not os.path.exists(fog_root):
        return False, "Missing VOC2007_CLEAN or VOC2007_FOG directories"
    
    # Check for images
    clean_imgs = os.path.join(clean_root, 'JPEGImages')
    fog_imgs = os.path.join(fog_root, 'JPEGImages')
    
    if not os.path.exists(clean_imgs) or not os.path.exists(fog_imgs):
        return False, "Missing JPEGImages directories"
    
    clean_count = len([f for f in os.listdir(clean_imgs) if f.endswith('.jpg')])
    fog_count = len([f for f in os.listdir(fog_imgs) if f.endswith('.jpg')])
    
    if clean_count == 0 or fog_count == 0:
        return False, f"No images found (clean: {clean_count}, fog: {fog_count})"
    
    if clean_count != fog_count:
        return False, f"Image count mismatch (clean: {clean_count}, fog: {fog_count})"
    
    return True, f"Dataset valid: {clean_count} paired images"

def main():
    parser = argparse.ArgumentParser(description='Setup VOC2007 synthetic fog training')
    parser.add_argument('--check', action='store_true', help='Check existing setup')
    parser.add_argument('--voc2007_root', type=str, help='Path to original VOC2007')
    parser.add_argument('--synfog_root', type=str, help='Path to synthetic fog dataset')
    parser.add_argument('--checkpoint', type=str, help='Path to RDFNet checkpoint')
    
    args = parser.parse_args()
    
    print("🔧 RDFNet VOC2007 Synthetic Fog Setup")
    print("=" * 50)
    
    if args.check:
        print("\n📋 Checking current setup...")
        
        # Check for training script
        if os.path.exists('voc2007_fog_train.py'):
            print("✅ Training script: voc2007_fog_train.py found")
        else:
            print("❌ Training script: voc2007_fog_train.py missing")
        
        # Check for dataset generation script
        if os.path.exists('tools/make_voc2007_fog_dataset.py'):
            print("✅ Dataset generation: tools/make_voc2007_fog_dataset.py found")
        else:
            print("❌ Dataset generation: tools/make_voc2007_fog_dataset.py missing")
        
        # Check synthetic fog dataset if specified
        if args.synfog_root:
            if os.path.exists(args.synfog_root):
                valid, msg = check_generated_dataset(args.synfog_root)
                if valid:
                    print(f"✅ Synthetic dataset: {msg}")
                else:
                    print(f"❌ Synthetic dataset: {msg}")
            else:
                print(f"❌ Synthetic dataset: {args.synfog_root} not found")
        
        # Check checkpoint if specified
        if args.checkpoint:
            if os.path.exists(args.checkpoint):
                print(f"✅ Checkpoint: {args.checkpoint} found")
            else:
                print(f"❌ Checkpoint: {args.checkpoint} not found")
    
    else:
        print("\n📋 Setup Instructions:")
        print("\n1. Generate Synthetic Fog Dataset:")
        if args.voc2007_root and args.synfog_root:
            missing = check_voc2007_structure(args.voc2007_root)
            if missing:
                print(f"   ❌ VOC2007 missing: {missing}")
                return
            else:
                print(f"   ✅ VOC2007 structure valid")
            
            print(f"   Command:")
            print(f"   python tools/make_voc2007_fog_dataset.py \\")
            print(f"       --voc2007_root {args.voc2007_root} \\")
            print(f"       --out_root {args.synfog_root}")
        else:
            print("   python tools/make_voc2007_fog_dataset.py \\")
            print("       --voc2007_root /path/to/VOC2007 \\") 
            print("       --out_root /path/to/VOC2007_SYNFOG")
        
        print("\n2. Update Training Script Paths:")
        print("   Edit voc2007_fog_train.py:")
        if args.synfog_root:
            print(f"   dataset_root = \"{args.synfog_root}\"")
        else:
            print("   dataset_root = \"/path/to/VOC2007_SYNFOG\"")
        
        if args.checkpoint:
            print(f"   checkpoint_path = \"{args.checkpoint}\"")
        else:
            print("   checkpoint_path = \"/path/to/RDFNet.pth\"")
        
        print("\n3. Run Training (Feature-Level Supervision):")
        print("   python voc2007_fog_train.py")
        print("   # Uses feature-level dehazing supervision:")
        print("   # L = L_detection + α_feat * warmup * Σ L1(F_l(restored), F_l(clean))")
        
        print("\n4. Monitor Results:")
        print("   - Checkpoints: /kaggle/working/logs/")
        print("   - TensorBoard logs: logs/loss_ours_feature_<timestamp>/")
        print("   - Experiment summary: experiment_summary_feature_supervision.json")
    
    print("\n" + "=" * 50)

if __name__ == '__main__':
    main()
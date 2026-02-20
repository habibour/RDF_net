#!/usr/bin/env python3
"""
Verify VOC2007 Paired Clean/Foggy Dataset Setup
"""

import os

def verify_paired_dataset():
    """Verify that clean and foggy image pairs exist."""
    
    dataset_root = "/Users/habibourakash/CSERUET/4-2/CSE4200-theisis/object_detection/VOC_FOG_12K_Upload"
    
    fog_dir = os.path.join(dataset_root, "VOC2007_FOG")
    clean_dir = os.path.join(dataset_root, "VOC2007_CLEAN", "JPEGImages")
    ann_dir = os.path.join(dataset_root, "VOC2007_Annotations")
    splits_dir = os.path.join(dataset_root, "ImageSets", "Main")
    
    print("🔍 VOC2007 Paired Dataset Verification")
    print("=" * 50)
    
    # Check directories
    dirs_to_check = [
        ("Fog images", fog_dir),
        ("Clean images", clean_dir), 
        ("Annotations", ann_dir),
        ("Splits", splits_dir)
    ]
    
    for name, path in dirs_to_check:
        exists = os.path.exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {name}: {path}")
        
        if exists and os.path.isdir(path):
            count = len([f for f in os.listdir(path) if f.endswith(('.jpg', '.xml', '.txt'))])
            print(f"    📊 Files: {count}")
    
    # Check splits
    split_files = ["train.txt", "val.txt", "test.txt", "trainval.txt"]
    print(f"\n📋 Dataset Splits:")
    
    for split_file in split_files:
        split_path = os.path.join(splits_dir, split_file)
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                lines = f.readlines()
            print(f"   ✅ {split_file}: {len(lines)} samples")
        else:
            print(f"   ❌ {split_file}: Missing")
    
    # Check sample pairs
    print(f"\n🔗 Checking Image Pairs:")
    
    # Get sample IDs from train split
    train_path = os.path.join(splits_dir, "train.txt")
    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            sample_ids = [line.strip() for line in f.readlines()[:5]]  # First 5
        
        for img_id in sample_ids:
            fog_img = os.path.join(fog_dir, f"{img_id}.jpg")
            clean_img = os.path.join(clean_dir, f"{img_id}.jpg")
            annotation = os.path.join(ann_dir, f"{img_id}.xml")
            
            fog_exists = os.path.exists(fog_img)
            clean_exists = os.path.exists(clean_img)  
            ann_exists = os.path.exists(annotation)
            
            if fog_exists and clean_exists and ann_exists:
                print(f"   ✅ {img_id}: Complete pair")
            else:
                missing = []
                if not fog_exists: missing.append("fog")
                if not clean_exists: missing.append("clean")
                if not ann_exists: missing.append("annotation")
                print(f"   ❌ {img_id}: Missing {', '.join(missing)}")
    
    print(f"\n🎯 Dataset ready for paired clean/foggy training!")
    print(f"📁 Dataset root: {dataset_root}")
    print(f"🚀 Run: python voc2007_paired_train.py")

if __name__ == '__main__':
    verify_paired_dataset()
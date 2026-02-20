#!/usr/bin/env python3
"""
Create 12k image subset with 80-10-10 split from VOC2007_SYNFOG dataset.
Generates train.txt, val.txt, test.txt files for RDFNet training.
"""

import os
import random
import shutil
from pathlib import Path

def create_12k_subset_with_split():
    """Create 12k image subset and generate 80-10-10 split."""
    
    # Paths
    synfog_root = "/Users/habibourakash/CSERUET/4-2/CSE4200-theisis/object_detection/VOC_FOG_12K_Upload"
    fog_images_dir = os.path.join(synfog_root, "VOC2007_FOG")
    imagesets_dir = os.path.join(synfog_root, "ImageSets", "Main")
    
    print("🎯 Creating 12k image subset with 80-10-10 split")
    print("=" * 50)
    
    # Get all image IDs
    image_files = [f for f in os.listdir(fog_images_dir) if f.endswith('.jpg')]
    image_ids = [os.path.splitext(f)[0] for f in image_files]
    
    total_available = len(image_ids)
    target_count = 12000
    
    print(f"📊 Available images: {total_available}")
    print(f"🎯 Target subset: {target_count}")
    
    if total_available < target_count:
        print(f"⚠️  Using all available {total_available} images (less than target 12k)")
        selected_ids = image_ids
    else:
        # Randomly select 12k images
        random.seed(42)  # Reproducible selection
        selected_ids = random.sample(image_ids, target_count)
        print(f"✅ Selected {len(selected_ids)} images randomly")
    
    # Create 80-10-10 split
    random.shuffle(selected_ids)  # Shuffle for random split
    
    n_total = len(selected_ids)
    n_train = int(0.8 * n_total)  # 80%
    n_val = int(0.1 * n_total)    # 10%
    n_test = n_total - n_train - n_val  # Remaining ~10%
    
    train_ids = selected_ids[:n_train]
    val_ids = selected_ids[n_train:n_train + n_val]
    test_ids = selected_ids[n_train + n_val:]
    
    print(f"\n📊 Dataset Split:")
    print(f"   🚂 Train: {len(train_ids)} images ({len(train_ids)/n_total*100:.1f}%)")
    print(f"   ✅ Val:   {len(val_ids)} images ({len(val_ids)/n_total*100:.1f}%)")
    print(f"   🔍 Test:  {len(test_ids)} images ({len(test_ids)/n_total*100:.1f}%)")
    
    # Create ImageSets directory if it doesn't exist
    os.makedirs(imagesets_dir, exist_ok=True)
    
    # Write split files
    splits = {
        'train.txt': train_ids,
        'val.txt': val_ids, 
        'test.txt': test_ids,
        'trainval.txt': train_ids + val_ids  # Combined for compatibility
    }
    
    for filename, ids in splits.items():
        filepath = os.path.join(imagesets_dir, filename)
        with open(filepath, 'w') as f:
            f.write('\n'.join(ids) + '\n')
        print(f"✅ Created: {filename} ({len(ids)} samples)")
    
    # Also create splits corresponding to clean dataset
    clean_imagesets_dir = os.path.join(synfog_root, "VOC2007_Annotations")
    os.makedirs(clean_imagesets_dir, exist_ok=True)
    
    for filename, ids in splits.items():
        filepath = os.path.join(clean_imagesets_dir, filename)
        with open(filepath, 'w') as f:
            f.write('\n'.join(ids) + '\n')
    
    print(f"✅ Split files created in both FOG and annotation directories")
    
    # Validate splits by checking file existence
    print(f"\n🔍 Validating splits...")
    fog_imgs = os.path.join(synfog_root, "VOC2007_FOG")
    annotations = os.path.join(synfog_root, "VOC2007_Annotations")
    
    missing_files = 0
    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        missing_in_split = 0
        for img_id in ids[:3]:  # Check first 3 of each split
            fog_img = os.path.join(fog_imgs, f"{img_id}.jpg")
            ann_file = os.path.join(annotations, f"{img_id}.xml")
            
            if not os.path.exists(fog_img):
                missing_in_split += 1
            if not os.path.exists(ann_file):
                missing_in_split += 1
        
        if missing_in_split == 0:
            print(f"   ✅ {split_name}: Files exist for sampled images")
        else:
            print(f"   ❌ {split_name}: Missing files detected")
            missing_files += missing_in_split
    
    if missing_files == 0:
        print(f"\n🎉 Dataset subset ready for training!")
        print(f"📁 Dataset path: {synfog_root}")
        print(f"📋 Split files: {imagesets_dir}")
        
        # Show training command
        print(f"\n🚀 Ready for training:")
        print(f"   1. Edit voc2007_fog_train.py:")
        print(f"      dataset_root = \"{synfog_root}\"")
        print(f"   2. Run: python voc2007_fog_train.py")
    else:
        print(f"\n⚠️  Validation found {missing_files} missing files")
    
    return synfog_root, len(selected_ids)

if __name__ == '__main__':
    dataset_path, total_images = create_12k_subset_with_split()
    print(f"\n✅ 12k subset creation complete: {total_images} images with 80-10-10 split")
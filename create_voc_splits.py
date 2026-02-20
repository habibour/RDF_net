#!/usr/bin/env python3
"""
Generate train/val/test splits for VOC_FOG_12K dataset
"""

import os
import random

# Set paths
fog_dir = "/Users/habibourakash/CSERUET/4-2/CSE4200-theisis/object_detection/VOC_FOG_12K_Upload/VOC2007_FOG"
output_dir = "/Users/habibourakash/CSERUET/4-2/CSE4200-theisis/object_detection/VOC_FOG_12K_Upload/ImageSets/Main"

# Get all image files and extract IDs
print("Scanning for fog images...")
image_ids = []
for filename in os.listdir(fog_dir):
    if filename.endswith('.jpg'):
        image_ids.append(filename[:-4])  # Remove .jpg extension

print(f"Found {len(image_ids)} fog images")

# Use first 12k images for the subset
target_count = min(12000, len(image_ids))
selected_ids = sorted(image_ids)[:target_count]  # Take first 12k in sorted order

print(f"Using {len(selected_ids)} images for training")

# Create 80-10-10 split
n_total = len(selected_ids)
n_train = int(0.8 * n_total)  # 9600
n_val = int(0.1 * n_total)    # 1200
n_test = n_total - n_train - n_val  # remaining ~1200

train_ids = selected_ids[:n_train]
val_ids = selected_ids[n_train:n_train + n_val]
test_ids = selected_ids[n_train + n_val:]

print(f"Split sizes: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")

# Create split files
splits = {
    'train.txt': train_ids,
    'val.txt': val_ids,
    'test.txt': test_ids,
    'trainval.txt': train_ids + val_ids
}

for split_name, ids in splits.items():
    split_path = os.path.join(output_dir, split_name)
    with open(split_path, 'w') as f:
        for img_id in ids:
            f.write(f"{img_id}\n")
    print(f"Created {split_name} with {len(ids)} samples")

# Validate splits
print("\nValidating splits...")
for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
    sample_ids = ids[:3]  # Check first 3
    missing = 0
    for img_id in sample_ids:
        fog_img = os.path.join(fog_dir, f"{img_id}.jpg")
        if not os.path.exists(fog_img):
            missing += 1
    
    if missing == 0:
        print(f"  ✅ {split_name}: Sample files exist")
    else:
        print(f"  ❌ {split_name}: {missing} files missing")

print(f"\n🎉 Dataset splits ready!")
print(f"📁 Location: {output_dir}")
print(f"📊 Total images: {len(selected_ids)}")
print(f"🚂 Train: {len(train_ids)} (80%)")
print(f"✅ Val: {len(val_ids)} (10%)")  
print(f"🔍 Test: {len(test_ids)} (10%)")
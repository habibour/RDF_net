import os
import random

# Configuration
fog_images_dir = "/Users/habibourakash/CSERUET/4-2/CSE4200-theisis/object_detection/VOC_FOG_12K_Upload/VOC2007_FOG"
output_dir = "/Users/habibourakash/CSERUET/4-2/CSE4200-theisis/object_detection/VOC_FOG_12K_Upload/ImageSets/Main"

# Get all image IDs
image_files = []
for filename in os.listdir(fog_images_dir):
    if filename.endswith('.jpg'):
        image_files.append(filename[:-4])

print(f"Found {len(image_files)} images")

# Select 12k images (or all if less than 12k)
target_count = min(12000, len(image_files))
random.seed(42)
selected_ids = random.sample(image_files, target_count)

print(f"Selected {len(selected_ids)} images")

# Create 80-10-10 split
random.shuffle(selected_ids)
n_total = len(selected_ids)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val

train_ids = selected_ids[:n_train]
val_ids = selected_ids[n_train:n_train + n_val]
test_ids = selected_ids[n_train + n_val:]

print(f"Train: {len(train_ids)}")
print(f"Val: {len(val_ids)}")
print(f"Test: {len(test_ids)}")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Write split files
splits = {
    'train.txt': train_ids,
    'val.txt': val_ids,
    'test.txt': test_ids,
    'trainval.txt': train_ids + val_ids
}

for filename, ids in splits.items():
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        for img_id in ids:
            f.write(f"{img_id}\n")
    print(f"Created {filename}: {len(ids)} samples")

print(f"\nSplit files created in: {output_dir}")
print("Dataset ready for training!")
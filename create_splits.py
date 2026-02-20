#!/usr/bin/env python3
"""
Create 80-10-10 train/val/test splits for VOC2012_FOGGY dataset
This script should be run in Kaggle after uploading your dataset
"""
import os
import random
import math

def create_voc_splits(dataset_root, output_dir="/kaggle/working", split_ratios=(0.8, 0.1, 0.1), seed=42):
    """
    Create train/val/test splits for VOC-format dataset
    
    Args:
        dataset_root: Path to VOC2012_FOGGY dataset
        output_dir: Directory to save split files
        split_ratios: Tuple of (train, val, test) ratios (should sum to 1.0)
        seed: Random seed for reproducible splits
    """
    print("🎯 Creating 80-10-10 Dataset Splits")
    print("=" * 60)
    
    # Validate split ratios
    if abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
    
    # Find images directory
    images_dir = os.path.join(dataset_root, "JPEGImages")
    annotations_dir = os.path.join(dataset_root, "Annotations")
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    if not os.path.exists(annotations_dir):
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    
    # Get all image files (without extensions)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_ids = [os.path.splitext(f)[0] for f in image_files]
    
    # Verify annotations exist
    valid_ids = []
    for img_id in image_ids:
        ann_path = os.path.join(annotations_dir, f"{img_id}.xml")
        if os.path.exists(ann_path):
            valid_ids.append(img_id)
    
    print(f"📊 Found {len(image_files)} images")
    print(f"📊 Found {len(valid_ids)} valid image-annotation pairs")
    
    if len(valid_ids) == 0:
        raise ValueError("No valid image-annotation pairs found!")
    
    # Shuffle for random splits
    random.seed(seed)
    random.shuffle(valid_ids)
    
    # Calculate split sizes
    total = len(valid_ids)
    train_size = int(total * split_ratios[0])
    val_size = int(total * split_ratios[1])
    test_size = total - train_size - val_size  # Remainder goes to test
    
    print(f"📊 Split sizes: train={train_size}, val={val_size}, test={test_size}")
    
    # Create splits
    train_ids = valid_ids[:train_size]
    val_ids = valid_ids[train_size:train_size + val_size]
    test_ids = valid_ids[train_size + val_size:]
    
    # Create output directory structure
    splits_dir = os.path.join(output_dir, "ImageSets", "Main")
    os.makedirs(splits_dir, exist_ok=True)
    
    # Write split files
    splits = {
        "train.txt": train_ids,
        "val.txt": val_ids, 
        "test.txt": test_ids
    }
    
    for split_name, ids in splits.items():
        split_path = os.path.join(splits_dir, split_name)
        with open(split_path, 'w') as f:
            for img_id in ids:
                f.write(f"{img_id}\n")
        print(f"✅ Created {split_name}: {len(ids)} samples -> {split_path}")
    
    return splits_dir

def create_usage_instructions():
    """Create instructions for using the splits"""
    instructions = """
# RDFNet Dataset Splits Usage Instructions

## 1. Run this script in Kaggle after uploading your dataset:
```python
python create_splits.py
```

## 2. The script will create:
- /kaggle/working/ImageSets/Main/train.txt (80% of data)
- /kaggle/working/ImageSets/Main/val.txt   (10% of data) 
- /kaggle/working/ImageSets/Main/test.txt  (10% of data)

## 3. Your kaggle_train.py will automatically find these splits in:
- /kaggle/working/ImageSets/Main/
- /kaggle/input/.../VOC2012_FOGGY/ImageSets/Main/ (if exists)

## 4. Dataset structure should be:
```
VOC2012_FOGGY/
├── JPEGImages/     # 12,000 foggy images
├── Annotations/    # 12,000 XML annotations
└── ImageSets/
    └── Main/
        ├── train.txt
        ├── val.txt 
        └── test.txt
```

## 5. Start training:
```python
python kaggle_train.py
```

The sanity checks will automatically verify your dataset structure!
"""
    return instructions

if __name__ == "__main__":
    print("🔧 VOC2012_FOGGY Dataset Splits Creator")
    print("=" * 60)
    
    # Example usage for Kaggle
    try:
        # Adjust this path to your actual dataset location in Kaggle
        dataset_path = "/kaggle/input/datasets/mdhabibourrahman/clean-foggy-images/training_voc_2012/VOC2012_FOGGY"
        
        if os.path.exists(dataset_path):
            print(f"📍 Found dataset at: {dataset_path}")
            splits_dir = create_voc_splits(dataset_path)
            print(f"\n🎉 Success! Splits created in: {splits_dir}")
        else:
            print(f"⚠️ Dataset not found at: {dataset_path}")
            print("📝 This is expected if running locally.")
            print("🔧 In Kaggle, update the dataset_path variable above.")
        
        print("\n📋 Usage Instructions:")
        print(create_usage_instructions())
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Make sure your dataset has the correct VOC structure:")
        print("   - JPEGImages/ directory with images")
        print("   - Annotations/ directory with XML files")
        
    print("=" * 60)
# RDFNet Kaggle Training Guide

This guide helps you run RDFNet training on Kaggle with the VOC_FOG_12K dataset.

## Quick Start (Recommended Approach)

```python
# Clone repo
!rm -rf RDF_net
!git clone https://github.com/habibour/RDF_net.git
%cd RDF_net

# Write voc_classes.txt
voc_classes = """aeroplane
bicycle
bird
boat
bottle
bus
car
cat
chair
cow
diningtable
dog
horse
motorbike
person
pottedplant
sheep
sofa
train
tvmonitor"""
with open('model_data/voc_classes.txt', 'w') as f:
    f.write(voc_classes)
print("Created voc_classes.txt")

# Set your dataset paths
VOC_SRC = "/kaggle/input/your-dataset-name/VOC_FOG_12K_Upload/VOC_FOG_12K_Upload"
RTTS_SRC = "/kaggle/input/your-dataset-name/RTTS/RTTS"
CHECKPOINT_PATH = "/kaggle/input/your-checkpoint-dataset/RDFNet.pth"

# Create expected structure via symlinks
!ln -s {VOC_SRC} /kaggle/working/VOC_FOG_12K_Upload
!ln -s {RTTS_SRC} /kaggle/working/RTTS

DATASET_ROOT = "/kaggle/working"

# Generate dataset splits
!python tools/make_vocfog_split.py --dataset_root "{DATASET_ROOT}/VOC_FOG_12K_Upload" --seed 42

# Fix dataset splits path (if training fails with FileNotFoundError)
!python fix_dataset_splits.py

# Update kaggle_train.py with your paths
import re
path = "kaggle_train.py"
with open(path, "r", encoding="utf-8") as f:
    data = f.read()

data = re.sub(r'dataset_root = ".*?"', f'dataset_root = "{DATASET_ROOT}"', data)
data = re.sub(r'checkpoint_path = ".*?"', f'checkpoint_path = "{CHECKPOINT_PATH}"', data)

with open(path, "w", encoding="utf-8") as f:
    f.write(data)

print("Updated kaggle_train.py with dataset_root and checkpoint_path")

# Verify splits were created
splits_dir = "/kaggle/working/dataset_splits/ImageSets/Main"
if os.path.exists(splits_dir):
    print("✅ Dataset splits created successfully!")
    for split_file in ["train.txt", "val.txt", "test.txt"]:
        split_path = os.path.join(splits_dir, split_file)
        if os.path.exists(split_path):
            with open(split_path, 'r') as f:
                count = len(f.readlines())
            print(f"   📄 {split_file}: {count} samples")
else:
    print("❌ Failed to create dataset splits")

# Start training
!python kaggle_train.py
```

## File Structure Expected

Your Kaggle input datasets should have this structure:

```
/kaggle/input/your-dataset-name/
├── VOC_FOG_12K_Upload/
│   └── VOC_FOG_12K_Upload/        # Actual dataset folder
│       ├── VOC2007_FOG/           # Foggy images from VOC2007
│       ├── VOC2007_Annotations/   # XML annotations for VOC2007
│       ├── VOC2012_FOG/           # Foggy images from VOC2012
│       └── VOC2012_Annotations/   # XML annotations for VOC2012
└── RTTS/
    └── RTTS/                      # Actual RTTS folder
        ├── JPEGImages/
        ├── Annotations/
        └── ImageSets/
```

After running the setup code, your working directory will have:

```
/kaggle/working/
├── RDF_net/                       # Cloned repository
├── VOC_FOG_12K_Upload/            # Symlink to input dataset
│   ├── VOC2007_FOG/
│   ├── VOC2007_Annotations/
│   ├── VOC2012_FOG/
│   ├── VOC2012_Annotations/
│   └── ImageSets/                 # Generated splits
│       └── Main/
│           ├── train.txt
│           ├── val.txt
│           └── test.txt
└── RTTS/                          # Symlink to RTTS dataset
```

## Configuration Options

### Training Methods

- `method_name = "baseline_pixel"` - Pixel-level dehazing
- `method_name = "ours_feature"` - Feature-level dehazing

### Key Parameters

- `training_epochs = 80` - Number of training epochs
- `save_period = 2` - Save checkpoint every 2 epochs
- `alpha_feat = 0.5` - Feature loss weight
- `lambda_pixel = 0.1` - Pixel loss weight

## Output Files

Training will create:

- `/kaggle/working/logs/` - Training logs and loss curves
- `/kaggle/working/checkpoints/` - Model checkpoints (downloadable)
- Experiment summary JSON with final results

## Troubleshooting

### Common Issues

1. **"Read-only file system" error**
   - Solution: The script now automatically creates splits in `/kaggle/working/`

2. **Missing packages (thop, colorama)**
   - Solution: Run `%run kaggle_setup.py` first

3. **Dataset path not found**
   - Check your dataset name in Kaggle and update `dataset_root`

4. **CUDA out of memory**
   - Reduce batch size: `Unfreeze_batch_size = 4`

### Memory Optimization

If you encounter memory issues:

```python
# In kaggle_train.py
Freeze_batch_size = 8      # Reduce from 16
Unfreeze_batch_size = 4    # Reduce from 8
num_workers = 1            # Reduce from 2
```

## Expected Runtime

- Dataset split generation: ~2-3 minutes
- Training (80 epochs): ~6-8 hours
- GPU memory usage: ~10-12GB

## Results

After training completes:

1. Download checkpoints from `/kaggle/working/checkpoints/`
2. Check experiment summary JSON for final mAP scores
3. Training curves available in logs directory

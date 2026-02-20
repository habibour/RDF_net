# RDFNet Training with Comprehensive Sanity Checks

## 🎯 Overview

This enhanced RDFNet training system includes **comprehensive sanity checks** and **centralized configuration** for robust fog-aware object detection training on Kaggle.

## ✅ Key Features

### 🔍 Comprehensive Sanity Checks

- **Directory existence validation** - Ensures all required dataset paths exist
- **Image count verification** - Validates ~12K fog/clean image pairs
- **Fog-clean pairing validation** - Critical for RDFNet feature-level supervision
- **RTTS isolation check** - Ensures RTTS is test-only (never used for training)
- **Model checkpoint validation** - Verifies pretrained weights availability
- **Sample image loading test** - Tests actual image loading functionality

### 📁 Centralized Configuration

- **Single source of truth** for all dataset paths
- **Kaggle-optimized paths** for your uploaded datasets
- **Easy configuration updates** without searching through code
- **Training method selection** (`baseline_pixel` vs `ours_feature`)

### 🛡️ Error Prevention

- **Automatic validation** before training starts
- **Clear error messages** with actionable fixes
- **Graceful failure** if dataset issues detected
- **Training safety** - prevents wasted GPU time on broken datasets

## 📊 Dataset Structure

```
Your Kaggle Input Data:
├── clean-foggy-images/training_voc_2012/
│   ├── VOC2012_FOGGY/
│   │   ├── JPEGImages/     # 12,000 synthetic foggy images
│   │   ├── Annotations/    # 12,000 XML annotations
│   │   └── ImageSets/Main/ # train.txt, val.txt, test.txt (80-10-10)
│   └── VOC2012_train_val/
│       ├── JPEGImages/     # 12,000 clean images (paired)
│       └── Annotations/    # 12,000 XML annotations (same as foggy)
├── foggy-voc/RTTS/
│   ├── JPEGImages/         # Real-world foggy images (test only)
│   └── Annotations/        # RTTS annotations (test only)
└── rdfnet-pth/
    └── RDFNet.pth         # Pretrained model weights
```

## 🚀 Quick Start Guide

### 1. Upload Your Dataset to Kaggle

- Upload your 12K paired fog/clean dataset
- Upload RTTS dataset for evaluation
- Upload RDFNet.pth pretrained weights

### 2. Create Dataset Splits (80-10-10)

```python
# Run in Kaggle notebook
!python create_splits.py
```

### 3. Configure Training Method

Edit `kaggle_train.py`:

```python
# TRAINING CONFIGURATION
method_name = "ours_feature"  # or "baseline_pixel"
epochs = 80
alpha_feat = 0.5              # Feature-level loss weight
lambda_pixel = 1.0            # Pixel-level loss weight
```

### 4. Start Training

```python
# Enable GPU in Kaggle settings first!
!python kaggle_train.py
```

The system will automatically:

1. ✅ Run comprehensive sanity checks
2. ✅ Validate dataset integrity
3. ✅ Check fog-clean pairing
4. ✅ Ensure RTTS isolation
5. 🚀 Start training if all checks pass

## 🔧 Configuration Details

### Dataset Paths (Centralized)

```python
# In kaggle_train.py - UPDATE THESE IF YOUR PATHS DIFFER
voc_fog_root = "/kaggle/input/datasets/mdhabibourrahman/clean-foggy-images/training_voc_2012/VOC2012_FOGGY"
voc_clean_root = "/kaggle/input/datasets/mdhabibourrahman/clean-foggy-images/training_voc_2012/VOC2012_train_val"
rtts_root = "/kaggle/input/datasets/mdhabibourrahman/foggy-voc/RTTS"
checkpoint_path = "/kaggle/input/datasets/mdhabibourrahman/rdfnet-pth/RDFNet.pth"
```

### Training Methods

#### Baseline Pixel-Level

```python
method_name = "baseline_pixel"
lambda_pixel = 1.0
# Only uses pixel-level restoration loss
```

#### Our Feature-Level (Recommended)

```python
method_name = "ours_feature"
lambda_pixel = 1.0
alpha_feat = 0.5
feat_warmup_epochs = 10
# Uses both pixel-level + feature-level supervision
```

## 🔍 Sanity Check Details

### What Gets Checked

1. **Essential directories exist**
   - VOC2012_FOGGY/JPEGImages & Annotations
   - VOC2012_train_val/JPEGImages & Annotations
   - RTTS/JPEGImages & Annotations

2. **Dataset size validation**
   - ~12,000 foggy images
   - ~12,000 clean images
   - Matching annotation counts

3. **Fog-clean pairing (CRITICAL for RDFNet)**
   - Each foggy image has corresponding clean pair
   - Annotation consistency between pairs
   - Sample validation (configurable sample size)

4. **RTTS isolation check**
   - RTTS never appears in train/val splits
   - RTTS used only for testing
   - Prevents data leakage

5. **Model checkpoint validation**
   - RDFNet.pth exists and is readable
   - File size reasonable (>10MB expected)

6. **Sample image loading**
   - PIL can load sample images
   - Image dimensions reasonable
   - No corrupted files in samples

### Sample Output

```
🔍 RUNNING COMPREHENSIVE DATASET SANITY CHECKS
============================================================
✅ Check 1: Essential directories...
   ✓ VOC Foggy JPEGImages: 12000 files
   ✓ VOC Foggy Annotations: 12000 files
   ✓ VOC Clean JPEGImages: 12000 files
   ✓ VOC Clean Annotations: 12000 files
✅ Check 2: Dataset size validation...
   ✓ Foggy images: 12000
   ✓ Clean images: 12000
   ✓ Foggy annotations: 12000
   ✓ Clean annotations: 12000
✅ Check 3: Fog-clean image pairing (checking 50 samples)...
   ✓ All 50 sample pairs are correctly matched
✅ Check 4: RTTS evaluation dataset...
   ✓ RTTS images: 4322
   ✓ RTTS annotations: 4322
✅ Check 5: Model checkpoint...
   ✓ Checkpoint found: RDFNet.pth (15.2 MB)
✅ Check 6: Sample image loading test...
   ✓ Fog image loaded: (500, 375) RGB
   ✓ Clean image loaded: (500, 375) RGB
============================================================
🎉 ALL SANITY CHECKS PASSED! Dataset is ready for training.
============================================================
```

## 📁 Files Overview

### Core Training Files

- **`kaggle_train.py`** - Main training script with integrated sanity checks
- **`create_splits.py`** - Creates 80-10-10 train/val/test splits
- **`test_config.py`** - Tests configuration without ML dependencies

### Configuration

- **Centralized paths** at top of `kaggle_train.py`
- **Training method selection** (`baseline_pixel` vs `ours_feature`)
- **Hyperparameter configuration** (learning rates, loss weights, etc.)

### Sanity Check System

- **`sanity_check_dataset()`** - Main validation function
- **Automatic execution** before training starts
- **Configurable validation depth** (sample sizes, checks to run)

## 🎯 Training Tips

### Method Selection

- **Use `baseline_pixel`** for comparison with pixel-only methods
- **Use `ours_feature`** for best RDFNet performance (recommended)
- **Feature warmup** gradually introduces feature-level loss

### GPU Requirements

- **Enable GPU** in Kaggle notebook settings
- **Mixed precision (fp16)** enabled by default for speed
- **CUDA required** - training will fail gracefully without GPU

### Monitoring

- **Loss curves** saved automatically to `/kaggle/working/logs/`
- **Model checkpoints** saved every 2 epochs to prevent loss
- **Backup checkpoints** to `/kaggle/working/checkpoints/`

## 🛠️ Troubleshooting

### Common Issues

#### Sanity Check Failures

```
❌ Missing VOC Foggy JPEGImages: /path/to/missing/directory
```

**Fix:** Check your Kaggle dataset upload paths match configuration

#### Pairing Issues

```
❌ Missing clean pair for fog image: image_001.jpg
```

**Fix:** Ensure fog/clean datasets have matching image IDs

#### RTTS in Training Data

```
❌ RTTS data detected in train/val annotations. RTTS must be test-only.
```

**Fix:** Check your train/val splits don't include RTTS paths

### Getting Help

1. **Check sanity check output** - detailed error messages
2. **Verify dataset structure** matches expected format
3. **Test configuration** with `test_config.py`
4. **Check file paths** in centralized configuration

## 📈 Expected Results

### Training Time

- **80 epochs** ~6-8 hours on Kaggle GPU
- **Automatic checkpointing** prevents data loss
- **Early validation** after 10 epochs

### Model Performance

- **Feature-level method** typically outperforms pixel-only
- **mAP improvements** on foggy object detection
- **Dehazing quality** measured by pixel-level loss

---

## 🎉 Ready to Train!

Your RDFNet training system is now equipped with:

- ✅ **Comprehensive validation** - catches issues before they waste GPU time
- ✅ **Centralized configuration** - easy to update and maintain
- ✅ **Robust error handling** - clear messages for quick fixes
- ✅ **Kaggle optimization** - paths and settings tuned for Kaggle environment

**Happy training! 🚀**

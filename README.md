### 📖 RDFNet: Real-time Object Detection Framework for Foggy Scenes

<a href="https://ieeexplore.ieee.org/document/11209981" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%93%9A Paper-IEEE-blue"></a>&ensp;
<a href="https://huggingface.co/spaces/PolarisFTL/RDFNet" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demos-blue"></a>&ensp;
![visitors](https://visitor-badge.laobi.icu/badge?page_id=PolarisFTL.RDFNet) <br />

[Tianle Fang](https://polarisftl.github.io/), Zhenbing Liu, Yutao Tang, Yingxin Huang, Haoxiang Lu, and [Chuangtao Zheng](https://github.com/15989715465) <br />
Computer Science and Information Security, Guilin University of Electronic Technology

---

![network](https://github.com/PolarisFTL/RDFNet/blob/main/figs/network.png)
_The architecture of the proposed RDFNet consists of several key components. Initially, the backbone extracts features from the input image. These extracted features are then processed through multiple branches that enter the neck and the LMDNet for multi-scale fusion and dehazing constraints. Subsequently, the feature maps pass through the head and are fed into the detection head to obtain the predicted targets. The generation of the foggy image is based on the ASM technique._

#### 😶‍🌫️ Experiments

![](https://github.com/PolarisFTL/RDFNet/blob/main/figs/result.png)
![](https://github.com/PolarisFTL/RDFNet/blob/main/figs/visual.png)

## 🔧 VOC2007 Synthetic Fog Training (New!)

### Generate Synthetic Fog Dataset

Create a paired fog/clean dataset from VOC2007 for training:

```bash
python tools/make_voc2007_fog_dataset.py \
    --voc2007_root /path/to/VOCdevkit/VOC2007 \
    --out_root /data/VOC2007_SYNFOG \
    --seed 42 \
    --beta_min 0.6 \
    --beta_max 2.2 \
    --A_min 0.75 \
    --A_max 1.0
```

**Parameters:**

- `--voc2007_root`: Path to VOC2007 directory (must contain JPEGImages, Annotations, ImageSets)
- `--out_root`: Output directory for synthetic fog dataset
- `--beta_min/max`: Scattering coefficient range for fog density
- `--A_min/max`: Atmospheric light range

**Generated Structure:**

```
VOC2007_SYNFOG/
├── VOC2007_CLEAN/
│   ├── JPEGImages/      # Original clean images
│   ├── Annotations/     # XML annotation files
│   └── ImageSets/       # Train/val/test splits
├── VOC2007_FOG/
│   ├── JPEGImages/      # Synthetic fog images
│   ├── Annotations/     # Same XML files as clean
│   └── ImageSets/       # Same splits as clean
├── README.md            # Dataset documentation
└── fog_parameters.json  # Fog generation parameters
```

### Train on Synthetic Fog Dataset

Fine-tune RDFNet from checkpoint using **feature-level dehazing supervision**:

```bash
# Edit voc2007_fog_train.py to set paths:
dataset_root = "/data/VOC2007_SYNFOG"
checkpoint_path = "/path/to/RDFNet.pth"

# Train with feature-level supervision
python voc2007_fog_train.py
```

**Training Method - Feature-Level Supervision:**

- **Loss Formulation**: `L = L_detection + α_feat * warmup * Σ L1(F_l(restored), F_l(clean))`
- **Feature Extraction**: Uses detector backbone features (P3/P4/P5 levels)
- **Supervision Strategy**: Matches features between restored and clean images
- **Warmup Schedule**: Gradual feature loss introduction over first N epochs
- **Advantages**: More robust than pixel-level supervision, focuses on detection-relevant features

**Key Parameters:**

```python
alpha_feat = 0.5        # Feature loss weight
feat_warmup_epochs = 10 # Warmup period for gradual feature loss introduction
training_epochs = 80    # Total fine-tuning epochs
```

### Expected Folder Structure

For Kaggle or local training:

```
workspace/
├── RDF_net/                    # This repository
├── VOC2007_SYNFOG/            # Generated dataset
│   ├── VOC2007_CLEAN/
│   └── VOC2007_FOG/
└── checkpoints/
    └── RDFNet.pth             # Pre-trained checkpoint
```

### Example Commands

Complete workflow from VOC2007:

```bash
# 1. Generate synthetic fog dataset
python tools/make_voc2007_fog_dataset.py \
    --voc2007_root /data/VOCdevkit/VOC2007 \
    --out_root /data/VOC2007_SYNFOG

# 2. Train with feature-level supervision (80 epochs)
# Edit voc2007_fog_train.py to set dataset_root and checkpoint_path
python voc2007_fog_train.py

# 3. Check results in experiment_summary_feature_supervision.json
```

### Output Files

Training generates:

- `logs/loss_ours_feature_<timestamp>/`: TensorBoard logs and evaluation results
- `logs/ep*-ours_feature.pth`: Checkpoint files every 2 epochs
- `logs/best_model.pth`: Best model based on validation mAP
- `experiment_summary_feature_supervision.json`: Training summary and final metrics

#### 📢News

<ul>
<li>November, 2024: Submitted paper.
<li>February, 19 2025: Rebuttal.
<li>March, 21 2025: Accept!
<li>June, 6 2025: Attended the meeting.
<li>December, 2 2025: The code has been uploaded.
</ul>

#### 🔧 Requirements and Installation

> - Python 3.9.0
> - PyTorch 1.10.0
> - Cudatoolkit 11.3
> - Numpy 1.25.1
> - Opencv-python 4.7.0.72

#### 👽 Installation

```
# Clone the RDFNet
git clone https://github.com/PolarisFTL/RDFNet.git
# Install dependent packages
cd RDFNet
```

#### 🚗 Datasets

| Dataset Name | Total Images | Train Set | Test Set | Google Drive | BaiduYun |
| ------------ | ------------ | --------- | -------- | ------------ | -------- |
| VOC-FOG      | 11,707       | 9,578     | 2,129    |              | —        |
| RTTS         | 4,322        | —         | 4,322    | —            | —        |
| FDD          | 101          | —         | 101      | —            | —        |

Organizing...

## 📊 Class Statistics

| Dataset | Images | Bicycle | Bus   | Car    | Motorbike | Person | All Objects |
| ------- | ------ | ------- | ----- | ------ | --------- | ------ | ----------- |
| VOC-FOG | 11,707 | 753     | 638   | 2,105  | 763       | 17,464 | 21,723      |
| RTTS    | 4,322  | 534     | 1,838 | 18,415 | 862       | 7,950  | 29,599      |
| FDD     | 101    | 17      | 17    | 425    | 9         | 269    | 737         |

#### 😺 Checkpoint

| Name   | Google                                                                                     | BaiduYun                                                       |
| ------ | ------------------------------------------------------------------------------------------ | -------------------------------------------------------------- |
| RDFNet | [Link](https://drive.google.com/file/d/1bXp9dWEX-XdVtqFNtwm6aKP3CVndxeES/view?usp=sharing) | [Link (1234)](https://pan.baidu.com/s/17YJJ6EA_5NfRTC7064bPgg) |

#### 🎈 Training and Testing

```python
# train RDFNet for VOC-FOG dataset
python tools/voc_annotations.py
# VOCdevkit_path='the path of VOC-FOG dataset' This step will generate the train.txt
# Then you need to modify the "JPEGImages" to "FOG", generating the train_fog.txt
modify the config.py
python train.py
# during training, the result will be saved in the logs
```

```python
# eval RDFNet for RTTS dataset
python get_map.py
# data_name='rtts,
# vocdevkit_path='the path of RTTS dataset'
# model_path = 'los/best_epoch_weights.pth'

python predict.py
# try to predict the image in fog weather
```

#### 🔥Model Performance

| Dataset | Params | FLOPs | FPS | mAP (%) |
| ------- | ------ | ----- | --- | ------- |
| VOC-FOG | 5.4M   | 13.7G | 115 | 78.39   |
| RTTS    | 5.4M   | 13.7G | 115 | 59.93   |
| FDD     | 5.4M   | 13.7G | 115 | 36.99   |

#### 🔗Citation

If this work is helpful for your research, please consider citing the following BibTeX entry.

```
@INPROCEEDINGS{11209981,
  author={Fang, Tianle and Liu, Zhenbing and Tang, Yutao and Huang, Yingxin and Lu, Haoxiang and Zheng, Chuangtao},
  booktitle={2025 IEEE International Conference on Multimedia and Expo (ICME)},
  title={RDFNet: Real-time Object Detection Framework for Foggy Scenes},
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Source coding;Computational modeling;Object detection;Detectors;Computer architecture;Multitasking;Feature extraction;Real-time systems;Complexity theory;Design optimization;Object detection;real-time detection;multi-task learning},
  doi={10.1109/ICME59968.2025.11209981}}
```

#### 📨 Contact

If you have any questions, please feel free to reach me out at polarisftl123@gmail.com

#### 🌻 Acknowledgement

This code is based on [YOLOv7-Tiny](https://github.com/bubbliiiing/yolov7-tiny-pytorch.git). Thanks for the awesome work.

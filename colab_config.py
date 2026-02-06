# ============================================================
# RDFNet Colab Configuration
# ============================================================
# This config is designed for Google Colab training with resume support

import os

# ============= DATASET PATHS (COLAB) =============
# Training data paths
TRAIN_FOG_PATH = '/content/drive/MyDrive/dataset/training/VOC2007 2/FOG'
TRAIN_ANN_PATH = '/content/drive/MyDrive/dataset/training/VOC2007 2/Annotations'

# Testing data paths (RTTS)
TEST_IMG_PATH = '/content/drive/MyDrive/dataset/RTTS/VOC2007/JPEGImages'
TEST_ANN_PATH = '/content/drive/MyDrive/dataset/RTTS/VOC2007/Annotations'

# Checkpoint directory on Drive (for resume)
CHECKPOINT_DIR = '/content/drive/MyDrive/RDFNet_training_checkpoints'

# ============= CLASSES =============
# VOC 20 classes (for training)
VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", 
               "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", 
               "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# RTTS 5 classes (for testing)
RTTS_CLASSES = ['bicycle', 'bus', 'car', 'motorbike', 'person']

# ============= MODEL =============
Cuda = True
seed = 114514
distributed = False
sync_bn = False
fp16 = True

# Use pretrained backbone for fresh training
model_path = 'model_data/yolov7_tiny_weights.pth'
classes_path = 'model_data/voc_classes.txt'
anchors_path = 'model_data/yolo_anchors.txt'
anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
input_shape = [640, 640]
pretrained = False

# ============= TRAINING EPOCHS =============
Init_Epoch = 0
Freeze_Epoch = 100
Freeze_batch_size = 16
UnFreeze_Epoch = 300
Unfreeze_batch_size = 8
Freeze_Train = True

# ============= OPTIMIZER =============
Init_lr = 1e-2
Min_lr = Init_lr * 0.01
optimizer_type = "sgd"
momentum = 0.937
weight_decay = 5e-4

# ============= LEARNING RATE =============
lr_decay_type = "cos"

# ============= SAVE =============
save_period = 10
save_dir = 'logs'
eval_flag = True
eval_period = 10

# ============= DATALOADER =============
num_workers = 4

# ============= ANNOTATION PATHS =============
train_annotation_path = '2007_train.txt'
val_annotation_path = '2007_val.txt'

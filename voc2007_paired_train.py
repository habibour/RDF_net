#!/usr/bin/env python3
"""
VOC2007 Paired Clean/Foggy Training Script for RDFNet
Trains with feature-level dehazing supervision using paired clean and foggy images.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# Add RDFNet to path
import sys
sys.path.append('/Users/habibourakash/CSERUET/4-2/CSE4200-theisis/object_detection/RDFNet')

from nets.yolo import YoloBody
from nets.model import RDFNet
from utils.dataloader import YoloDataset, yolo_dataset_collate_fn
from utils.utils_fit import fit_one_epoch
from utils.utils import get_anchors, get_classes
from utils.callbacks import EvalCallback

class PairedFogCleanDataset(YoloDataset):
    """
    Dataset for paired clean and foggy images training.
    """
    def __init__(self, annotation_lines, input_shape, num_classes, train, 
                 fog_root, clean_root):
        super().__init__(annotation_lines, input_shape, num_classes, train)
        self.fog_root = fog_root
        self.clean_root = clean_root
    
    def __getitem__(self, index):
        # Get foggy image and annotations (primary)
        fog_image, box = super().__getitem__(index)
        
        # Get corresponding clean image
        annotation_line = self.annotation_lines[index]
        line = annotation_line.split()
        
        # Extract image ID from annotation
        fog_path = line[0]
        img_id = os.path.basename(fog_path).replace('.jpg', '')
        
        # Load clean image
        clean_path = os.path.join(self.clean_root, f"{img_id}.jpg")
        clean_image = self.get_random_data(clean_path, self.input_shape[0:2], 
                                         random=self.train, jitter=0.3, 
                                         hue=0.1, sat=0.7, val=0.4)[0]
        
        return fog_image, clean_image, box


def get_voc2007_fog_annotation_lines(dataset_root):
    """
    Generate annotation lines for VOC2007 fog dataset.
    """
    fog_images_dir = os.path.join(dataset_root, 'VOC2007_FOG')
    annotations_dir = os.path.join(dataset_root, 'VOC2007_Annotations') 
    
    annotation_lines = []
    
    # Get image IDs from fog directory
    for img_file in os.listdir(fog_images_dir):
        if img_file.endswith('.jpg'):
            img_id = img_file[:-4]
            
            fog_path = os.path.join(fog_images_dir, img_file)
            ann_path = os.path.join(annotations_dir, f"{img_id}.xml")
            
            if os.path.exists(ann_path):
                # Parse XML annotation
                import xml.etree.ElementTree as ET
                tree = ET.parse(ann_path)
                root = tree.getroot()
                
                boxes = []
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    if name in ['person', 'car', 'bicycle', 'motorbike', 'bus', 'train']:
                        # Map to simplified classes
                        class_id = {'person': 0, 'car': 1, 'bicycle': 2, 
                                  'motorbike': 3, 'bus': 4, 'train': 5}[name]
                        
                        bbox = obj.find('bndbox')
                        xmin = int(bbox.find('xmin').text)
                        ymin = int(bbox.find('ymin').text) 
                        xmax = int(bbox.find('xmax').text)
                        ymax = int(bbox.find('ymax').text)
                        
                        boxes.append(f"{xmin},{ymin},{xmax},{ymax},{class_id}")
                
                if boxes:
                    line = fog_path + ' ' + ' '.join(boxes)
                    annotation_lines.append(line)
    
    return annotation_lines


def main():
    print("🎯 Starting VOC2007 Paired Clean/Foggy Training")
    print("=" * 60)
    
    # Configuration
    dataset_root = "/Users/habibourakash/CSERUET/4-2/CSE4200-theisis/object_detection/VOC_FOG_12K_Upload"
    fog_images_root = os.path.join(dataset_root, "VOC2007_FOG")
    clean_images_root = os.path.join(dataset_root, "VOC2007_CLEAN", "JPEGImages") 
    
    input_shape = (640, 640)
    num_classes = 6  # Simplified classes
    batch_size = 8
    lr = 1e-4
    epochs = 80
    
    # Model paths
    model_path = "/Users/habibourakash/CSERUET/4-2/CSE4200-theisis/object_detection/ep060-loss0.123.pth"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Validate dataset structure
    print(f"\n📁 Dataset validation:")
    print(f"   Fog images: {fog_images_root}")
    print(f"   Clean images: {clean_images_root}")
    print(f"   Fog images exist: {os.path.exists(fog_images_root)}")
    print(f"   Clean images exist: {os.path.exists(clean_images_root)}")
    
    fog_count = len([f for f in os.listdir(fog_images_root) if f.endswith('.jpg')])
    clean_count = len([f for f in os.listdir(clean_images_root) if f.endswith('.jpg')])
    print(f"   Fog image count: {fog_count}")
    print(f"   Clean image count: {clean_count}")
    
    # Load dataset splits
    splits_dir = os.path.join(dataset_root, "ImageSets", "Main")
    train_split = os.path.join(splits_dir, "train.txt")
    val_split = os.path.join(splits_dir, "val.txt")
    
    if not os.path.exists(train_split):
        print(f"❌ Train split not found: {train_split}")
        return
        
    # Generate annotation lines
    print(f"\n📋 Generating annotations...")
    all_annotation_lines = get_voc2007_fog_annotation_lines(dataset_root)
    print(f"   Total annotations: {len(all_annotation_lines)}")
    
    # Load splits
    with open(train_split, 'r') as f:
        train_ids = [line.strip() for line in f.readlines()]
    with open(val_split, 'r') as f:
        val_ids = [line.strip() for line in f.readlines()]
    
    # Filter annotations by split
    train_lines = [line for line in all_annotation_lines 
                   if any(tid in line for tid in train_ids)]
    val_lines = [line for line in all_annotation_lines 
                 if any(vid in line for vid in val_ids)]
    
    print(f"   Train samples: {len(train_lines)}")
    print(f"   Val samples: {len(val_lines)}")
    
    # Initialize model
    print(f"\n🏗️  Initializing RDFNet model...")
    anchors = get_anchors()
    model = RDFNet(anchors, num_classes, pretrained=False)
    
    # Load checkpoint
    if os.path.exists(model_path):
        print(f"📦 Loading checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    
    # Create datasets
    print(f"\n📊 Creating paired datasets...")
    train_dataset = PairedFogCleanDataset(
        train_lines, input_shape, num_classes, True,
        fog_images_root, clean_images_root
    )
    val_dataset = PairedFogCleanDataset(
        val_lines, input_shape, num_classes, False,
        fog_images_root, clean_images_root  
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=yolo_dataset_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=yolo_dataset_collate_fn
    )
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss function - Combined detection + feature matching
    yolo_loss = nn.MSELoss()
    feature_loss = nn.L1Loss()
    
    print(f"\n🚀 Starting training for {epochs} epochs...")
    print(f"   Method: Feature-level dehazing supervision")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    
    # Training loop
    best_loss = float('inf')
    save_dir = "/Users/habibourakash/CSERUET/4-2/CSE4200-theisis/object_detection/paired_training_logs"
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        print(f"\n📈 Epoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            if len(batch_data) == 3:  # fog_image, clean_image, targets
                fog_images, clean_images, targets = batch_data
                fog_images = fog_images.to(device)
                clean_images = clean_images.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass on foggy images  
                fog_outputs = model(fog_images)
                
                # Forward pass on clean images
                with torch.no_grad():
                    clean_outputs = model(clean_images)
                
                # Detection loss (on foggy images)
                detection_loss = yolo_loss(fog_outputs[0], targets)
                
                # Feature matching loss
                feat_loss = 0.0
                if len(fog_outputs) > 1 and len(clean_outputs) > 1:
                    for f_feat, c_feat in zip(fog_outputs[1:], clean_outputs[1:]):
                        feat_loss += feature_loss(f_feat, c_feat)
                
                # Combined loss
                total_loss = detection_loss + 0.1 * feat_loss
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                num_batches += 1
                
                if batch_idx % 50 == 0:
                    print(f"   Batch {batch_idx}: Loss={total_loss.item():.4f}")
        
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                if len(batch_data) == 3:
                    fog_images, clean_images, targets = batch_data
                    fog_images = fog_images.to(device)
                    clean_images = clean_images.to(device)
                    
                    fog_outputs = model(fog_images)
                    clean_outputs = model(clean_images)
                    
                    val_loss += yolo_loss(fog_outputs[0], targets).item()
                    val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        print(f"📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}")
        print(f"   LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = os.path.join(save_dir, f"best_paired_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best model: {save_path}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 20 == 0:
            save_path = os.path.join(save_dir, f"paired_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved checkpoint: {save_path}")
        
        scheduler.step()
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Models saved in: {save_dir}")
    print(f"🏆 Best validation loss: {best_loss:.4f}")

if __name__ == '__main__':
    main()
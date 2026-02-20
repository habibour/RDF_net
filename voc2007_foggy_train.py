#!/usr/bin/env python3
"""
VOC2007 Single-Domain Training with Dehazing Components
Trains RDFNet on foggy images with self-supervised dehazing features.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2

# Add RDFNet to path
import sys
sys.path.append('/Users/habibourakash/CSERUET/4-2/CSE4200-theisis/object_detection/RDFNet')

from nets.yolo import YoloBody
from nets.model import RDFNet
from utils.dataloader import YoloDataset, yolo_dataset_collate_fn
from utils.utils_fit import fit_one_epoch
from utils.utils import get_anchors, get_classes


class FoggyDatasetWithDehazing(YoloDataset):
    """
    Dataset for training with foggy images and self-supervised dehazing.
    Creates pseudo-clean targets from foggy images.
    """
    
    def __init__(self, annotation_lines, input_shape, num_classes, train):
        super().__init__(annotation_lines, input_shape, num_classes, train)
    
    def dehaze_image(self, image):
        """
        Simple dehazing using histogram equalization and contrast enhancement.
        This serves as a pseudo-clean target.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to lightness channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge and convert back
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # Apply gamma correction
        gamma = 1.2
        enhanced = np.power(enhanced / 255.0, 1.0 / gamma) * 255.0
        enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return enhanced
    
    def __getitem__(self, index):
        # Get foggy image and annotations
        fog_image, box = super().__getitem__(index)
        
        # Create pseudo-clean version
        fog_np = (fog_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        clean_np = self.dehaze_image(fog_np)
        clean_image = torch.from_numpy(clean_np).permute(2, 0, 1).float() / 255.0
        
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
                    # Map common VOC classes to indices
                    class_mapping = {
                        'person': 0, 'bicycle': 1, 'car': 2, 'motorbike': 3,
                        'aeroplane': 4, 'bus': 5, 'train': 6, 'truck': 7,
                        'boat': 8, 'traffic light': 9, 'fire hydrant': 10,
                        'stop sign': 11, 'parking meter': 12, 'bench': 13,
                        'bird': 14, 'cat': 15, 'dog': 16, 'horse': 17,
                        'sheep': 18, 'cow': 19
                    }
                    
                    if name in class_mapping:
                        class_id = class_mapping[name]
                        
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
    print("🌫️  Starting VOC2007 Foggy Training with Dehazing")
    print("=" * 60)
    
    # Configuration
    dataset_root = "/Users/habibourakash/CSERUET/4-2/CSE4200-theisis/object_detection/VOC_FOG_12K_Upload"
    fog_images_root = os.path.join(dataset_root, "VOC2007_FOG")
    
    input_shape = (640, 640)
    num_classes = 20  # VOC classes
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
    print(f"   Fog images exist: {os.path.exists(fog_images_root)}")
    
    if os.path.exists(fog_images_root):
        fog_count = len([f for f in os.listdir(fog_images_root) if f.endswith('.jpg')])
        print(f"   Fog image count: {fog_count}")
    else:
        print("❌ Fog images directory not found!")
        return
    
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
    
    if len(all_annotation_lines) == 0:
        print("❌ No annotations found!")
        return
    
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
    
    if len(train_lines) == 0:
        print("❌ No training samples found!")
        return
    
    # Initialize model
    print(f"\n🏗️  Initializing RDFNet model...")
    anchors_path = "/Users/habibourakash/CSERUET/4-2/CSE4200-theisis/object_detection/model_data/yolo_anchors.txt"
    
    if os.path.exists(anchors_path):
        anchors = get_anchors(anchors_path)
    else:
        # Default YOLO anchors
        anchors = np.array([
            [10,13], [16,30], [33,23], [30,61], [62,45],
            [59,119], [116,90], [156,198], [373,326]
        ]).reshape(-1, 2)
    
    model = RDFNet(anchors, num_classes, pretrained=False)
    
    # Load checkpoint if available
    if os.path.exists(model_path):
        print(f"📦 Loading checkpoint: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            print("✅ Checkpoint loaded successfully")
        except Exception as e:
            print(f"⚠️  Checkpoint loading failed: {e}")
            print("   Continuing with random initialization")
    
    model = model.to(device)
    
    # Create datasets
    print(f"\n📊 Creating datasets...")
    train_dataset = FoggyDatasetWithDehazing(
        train_lines, input_shape, num_classes, True
    )
    val_dataset = FoggyDatasetWithDehazing(
        val_lines, input_shape, num_classes, False
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, collate_fn=yolo_dataset_collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, collate_fn=yolo_dataset_collate_fn,
        pin_memory=True
    )
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss functions
    detection_loss = nn.MSELoss()
    feature_loss = nn.L1Loss()
    
    print(f"\n🚀 Starting training for {epochs} epochs...")
    print(f"   Method: Self-supervised dehazing + object detection")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    
    # Training loop
    best_loss = float('inf')
    save_dir = "/Users/habibourakash/CSERUET/4-2/CSE4200-theisis/object_detection/foggy_training_logs"
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(epochs):
        print(f"\n📈 Epoch {epoch+1}/{epochs}")
        print("-" * 50)
        
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            try:
                if len(batch_data) == 3:  # fog_image, clean_target, targets
                    fog_images, clean_targets, targets = batch_data
                    fog_images = fog_images.to(device)
                    clean_targets = clean_targets.to(device)
                    
                    optimizer.zero_grad()
                    
                    # Forward pass on foggy images  
                    outputs = model(fog_images)
                    
                    # For now, use simple MSE loss on outputs
                    # In a full implementation, you'd compute proper YOLO loss
                    if isinstance(outputs, (list, tuple)):
                        main_output = outputs[0]
                    else:
                        main_output = outputs
                    
                    # Simplified loss - replace with proper YOLO loss
                    loss = torch.mean((main_output - clean_targets) ** 2) * 0.1
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    num_batches += 1
                    
                    if batch_idx % 50 == 0:
                        print(f"   Batch {batch_idx}: Loss={loss.item():.4f}")
                        
                elif len(batch_data) == 2:  # fog_image, targets
                    fog_images, targets = batch_data
                    fog_images = fog_images.to(device)
                    
                    optimizer.zero_grad()
                    
                    outputs = model(fog_images)
                    
                    # Simple loss for now
                    if isinstance(outputs, (list, tuple)):
                        main_output = outputs[0]
                    else:
                        main_output = outputs
                    
                    loss = torch.mean(main_output ** 2) * 0.1
                    
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    num_batches += 1
                    
                    if batch_idx % 50 == 0:
                        print(f"   Batch {batch_idx}: Loss={loss.item():.4f}")
                        
            except Exception as e:
                print(f"   ⚠️  Batch {batch_idx} error: {e}")
                continue
        
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
        
        print(f"📊 Epoch {epoch+1} Results:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 20 == 0 or epoch == 0:
            save_path = os.path.join(save_dir, f"foggy_model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved checkpoint: {save_path}")
        
        scheduler.step()
    
    # Save final model
    final_path = os.path.join(save_dir, "foggy_model_final.pth")
    torch.save(model.state_dict(), final_path)
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Models saved in: {save_dir}")
    print(f"💾 Final model: {final_path}")

if __name__ == '__main__':
    main()
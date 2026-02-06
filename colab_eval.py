"""
RDFNet Evaluation Script for RTTS Dataset
"""

import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

from yolo import YOLO
from utils.utils_map import get_map
from colab_config import CHECKPOINT_DIR, TEST_IMG_PATH, TEST_ANN_PATH, RTTS_CLASSES


def find_best_checkpoint():
    """Find the best or latest checkpoint"""
    search_paths = ['logs', CHECKPOINT_DIR]
    
    for path in search_paths:
        if not os.path.exists(path):
            continue
        
        files = os.listdir(path)
        
        # Prefer 'best' checkpoint
        best_files = [f for f in files if 'best' in f.lower() and f.endswith('.pth')]
        if best_files:
            return os.path.join(path, best_files[0])
        
        # Otherwise find latest epoch
        epoch_files = [f for f in files if f.startswith('ep') and f.endswith('.pth')]
        if epoch_files:
            epoch_files.sort(key=lambda x: int(x.split('-')[0][2:]), reverse=True)
            return os.path.join(path, epoch_files[0])
    
    # Fallback to pretrained model
    if os.path.exists('model_data/RDFNet.pth'):
        return 'model_data/RDFNet.pth'
    
    return None


def evaluate_rtts():
    """Evaluate model on RTTS dataset"""
    print("=" * 60)
    print("RDFNet Evaluation on RTTS Dataset")
    print("=" * 60)
    
    # Find checkpoint
    model_path = find_best_checkpoint()
    if not model_path:
        print("❌ No checkpoint found!")
        return
    
    print(f"\n📦 Using checkpoint: {model_path}")
    
    # Verify RTTS paths
    if not os.path.exists(TEST_IMG_PATH):
        print(f"❌ RTTS images not found: {TEST_IMG_PATH}")
        return
    
    if not os.path.exists(TEST_ANN_PATH):
        print(f"❌ RTTS annotations not found: {TEST_ANN_PATH}")
        return
    
    # Initialize YOLO
    print("\n🔄 Loading model...")
    yolo = YOLO(
        model_path=model_path,
        classes_path='model_data/rtts_classes.txt',
        anchors_path='model_data/yolo_anchors.txt',
        input_shape=[640, 640],
        phi='l',
        confidence=0.5,
        nms_iou=0.3,
        cuda=True
    )
    
    # Create output directories
    os.makedirs('map_out/ground-truth', exist_ok=True)
    os.makedirs('map_out/detection-results', exist_ok=True)
    
    # Get test images
    xml_files = [f for f in os.listdir(TEST_ANN_PATH) if f.endswith('.xml')]
    print(f"\n📊 Processing {len(xml_files)} images...")
    
    for xml_file in tqdm(xml_files):
        img_id = xml_file[:-4]
        
        # Find image
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
            p = os.path.join(TEST_IMG_PATH, img_id + ext)
            if os.path.exists(p):
                img_path = p
                break
        
        if not img_path:
            continue
        
        # Ground truth
        tree = ET.parse(os.path.join(TEST_ANN_PATH, xml_file))
        root = tree.getroot()
        
        with open(f'map_out/ground-truth/{img_id}.txt', 'w') as f:
            for obj in root.iter('object'):
                name = obj.find('name').text.lower()
                if name not in RTTS_CLASSES:
                    continue
                bbox = obj.find('bndbox')
                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))
                f.write(f"{name} {xmin} {ymin} {xmax} {ymax}\n")
        
        # Detection
        try:
            image = Image.open(img_path)
            yolo.get_map_txt(img_id, image, RTTS_CLASSES, 'map_out')
        except Exception as e:
            print(f"Error processing {img_id}: {e}")
    
    # Calculate mAP
    print("\n📊 Calculating mAP...")
    print("=" * 60)
    get_map(0.5, True, path='map_out')
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("📁 Results saved to: map_out/")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_rtts()

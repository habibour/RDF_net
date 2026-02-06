"""
RDFNet Kaggle Evaluation Script - Evaluate on RTTS Dataset
"""
import os
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm

# ============= KAGGLE PATHS =============
RTTS_IMAGES = '/kaggle/input/foggy-voc/RTTS/RTTS/JPEGImages'
RTTS_ANNOTATIONS = '/kaggle/input/foggy-voc/RTTS/RTTS/Annotations'
OUTPUT_DIR = '/kaggle/working'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

RTTS_CLASSES = ['bicycle', 'bus', 'car', 'motorbike', 'person']


def find_best_checkpoint():
    """Find the best or latest checkpoint"""
    search_dirs = [
        os.path.join(OUTPUT_DIR, 'logs'),
        CHECKPOINT_DIR,
        OUTPUT_DIR
    ]
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        
        files = os.listdir(search_dir)
        
        # Look for best weights first
        for f in files:
            if 'best' in f.lower() and f.endswith('.pth'):
                return os.path.join(search_dir, f)
        
        # Otherwise find latest epoch
        epoch_files = [(f, int(f.split('-')[0][2:])) for f in files 
                       if f.startswith('ep') and f.endswith('.pth')]
        if epoch_files:
            epoch_files.sort(key=lambda x: x[1], reverse=True)
            return os.path.join(search_dir, epoch_files[0][0])
        
        # Last resort: last_epoch_weights
        if 'last_epoch_weights.pth' in files:
            return os.path.join(search_dir, 'last_epoch_weights.pth')
    
    return None


def evaluate_rtts(model_path=None):
    """Evaluate model on RTTS dataset"""
    from yolo import YOLO
    from utils.utils_map import get_map
    
    # Find checkpoint
    if model_path is None:
        model_path = find_best_checkpoint()
    
    if model_path is None or not os.path.exists(model_path):
        print("❌ No checkpoint found!")
        print("   Please train first or specify model_path")
        return
    
    print("=" * 60)
    print("📊 RDFNet Evaluation on RTTS")
    print("=" * 60)
    print(f"📦 Using: {model_path}")
    
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
    map_out_path = os.path.join(OUTPUT_DIR, 'map_out')
    os.makedirs(os.path.join(map_out_path, 'ground-truth'), exist_ok=True)
    os.makedirs(os.path.join(map_out_path, 'detection-results'), exist_ok=True)
    
    # Get test images
    xml_files = [f for f in os.listdir(RTTS_ANNOTATIONS) if f.endswith('.xml')]
    print(f"\n📊 Processing {len(xml_files)} images...")
    
    processed = 0
    for xml_file in tqdm(xml_files):
        img_id = xml_file[:-4]
        
        # Find image
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
            p = os.path.join(RTTS_IMAGES, img_id + ext)
            if os.path.exists(p):
                img_path = p
                break
        
        if not img_path:
            continue
        
        # Ground truth
        try:
            tree = ET.parse(os.path.join(RTTS_ANNOTATIONS, xml_file))
            root = tree.getroot()
            
            with open(os.path.join(map_out_path, 'ground-truth', f'{img_id}.txt'), 'w') as f:
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
            image = Image.open(img_path)
            yolo.get_map_txt(img_id, image, RTTS_CLASSES, map_out_path)
            processed += 1
            
        except Exception as e:
            print(f"Error: {img_id} - {e}")
    
    print(f"\n✅ Processed: {processed} images")
    
    # Calculate mAP
    print("\n" + "=" * 60)
    print("📊 Calculating mAP...")
    print("=" * 60)
    
    get_map(0.5, True, path=map_out_path)
    
    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print(f"📁 Results: {map_out_path}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_rtts()

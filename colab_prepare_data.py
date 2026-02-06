"""
Prepare training and testing annotations for Colab
This script generates annotation files from your Drive dataset
"""

import os
import xml.etree.ElementTree as ET
import random
from colab_config import (TRAIN_FOG_PATH, TRAIN_ANN_PATH, TEST_IMG_PATH, TEST_ANN_PATH,
                          VOC_CLASSES, RTTS_CLASSES)


def convert_voc_annotation(xml_path, classes):
    """Convert VOC XML annotation to YOLO format"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult')
            if difficult is not None and difficult.text == '1':
                continue
            name = obj.find('name').text
            if name not in classes:
                continue
            bbox = obj.find('bndbox')
            b = (int(float(bbox.find('xmin').text)), int(float(bbox.find('ymin').text)),
                 int(float(bbox.find('xmax').text)), int(float(bbox.find('ymax').text)))
            boxes.append(f"{b[0]},{b[1]},{b[2]},{b[3]},{classes.index(name)}")
        return boxes
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return []


def generate_training_annotations():
    """Generate training and validation annotation files"""
    print("=" * 60)
    print("Generating Training Annotations")
    print("=" * 60)
    
    if not os.path.exists(TRAIN_FOG_PATH):
        print(f"❌ Training FOG path not found: {TRAIN_FOG_PATH}")
        return False
    
    if not os.path.exists(TRAIN_ANN_PATH):
        print(f"❌ Training Annotations path not found: {TRAIN_ANN_PATH}")
        return False
    
    # Get valid images (have both FOG image and annotation)
    fog_files = set(os.listdir(TRAIN_FOG_PATH))
    xml_files = [f[:-4] for f in os.listdir(TRAIN_ANN_PATH) if f.endswith('.xml')]
    valid_ids = [x for x in xml_files if f"{x}.jpg" in fog_files]
    
    print(f"📁 FOG images: {len(fog_files)}")
    print(f"📁 XML annotations: {len(xml_files)}")
    print(f"✅ Valid pairs: {len(valid_ids)}")
    
    if len(valid_ids) == 0:
        print("❌ No valid image-annotation pairs found!")
        return False
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(valid_ids)
    train_split = int(len(valid_ids) * 0.9)
    train_ids = valid_ids[:train_split]
    val_ids = valid_ids[train_split:]
    
    # Write train annotations
    train_count = 0
    with open('2007_train.txt', 'w') as f:
        for img_id in train_ids:
            img_path = f"{TRAIN_FOG_PATH}/{img_id}.jpg"
            boxes = convert_voc_annotation(f"{TRAIN_ANN_PATH}/{img_id}.xml", VOC_CLASSES)
            if boxes:
                f.write(f"{img_path} {' '.join(boxes)}\n")
                train_count += 1
    
    # Write val annotations
    val_count = 0
    with open('2007_val.txt', 'w') as f:
        for img_id in val_ids:
            img_path = f"{TRAIN_FOG_PATH}/{img_id}.jpg"
            boxes = convert_voc_annotation(f"{TRAIN_ANN_PATH}/{img_id}.xml", VOC_CLASSES)
            if boxes:
                f.write(f"{img_path} {' '.join(boxes)}\n")
                val_count += 1
    
    print(f"\n✅ Train annotations: {train_count} images → 2007_train.txt")
    print(f"✅ Val annotations: {val_count} images → 2007_val.txt")
    
    # Create VOC classes file
    os.makedirs('model_data', exist_ok=True)
    with open('model_data/voc_classes.txt', 'w') as f:
        f.write('\n'.join(VOC_CLASSES))
    print("✅ Created model_data/voc_classes.txt")
    
    return True


def generate_rtts_annotations():
    """Generate RTTS test annotation file"""
    print("\n" + "=" * 60)
    print("Generating RTTS Test Annotations")
    print("=" * 60)
    
    if not os.path.exists(TEST_IMG_PATH):
        print(f"❌ RTTS images path not found: {TEST_IMG_PATH}")
        return False
    
    if not os.path.exists(TEST_ANN_PATH):
        print(f"❌ RTTS annotations path not found: {TEST_ANN_PATH}")
        return False
    
    test_count = 0
    with open('rtts_test.txt', 'w') as f:
        for xml_file in os.listdir(TEST_ANN_PATH):
            if not xml_file.endswith('.xml'):
                continue
            img_id = xml_file[:-4]
            
            # Find image
            img_file = None
            for ext in ['.jpg', '.png', '.jpeg', '.JPG', '.PNG']:
                if os.path.exists(os.path.join(TEST_IMG_PATH, img_id + ext)):
                    img_file = img_id + ext
                    break
            
            if img_file:
                boxes = convert_voc_annotation(os.path.join(TEST_ANN_PATH, xml_file), RTTS_CLASSES)
                if boxes:
                    f.write(f"{TEST_IMG_PATH}/{img_file} {' '.join(boxes)}\n")
                    test_count += 1
    
    print(f"✅ RTTS test annotations: {test_count} images → rtts_test.txt")
    
    # Create RTTS classes file
    os.makedirs('model_data', exist_ok=True)
    with open('model_data/rtts_classes.txt', 'w') as f:
        f.write('\n'.join(RTTS_CLASSES))
    print("✅ Created model_data/rtts_classes.txt")
    
    return True


if __name__ == "__main__":
    print("\n🚀 RDFNet Data Preparation\n")
    
    train_ok = generate_training_annotations()
    test_ok = generate_rtts_annotations()
    
    print("\n" + "=" * 60)
    if train_ok and test_ok:
        print("✅ All annotations generated successfully!")
        print("\nNext steps:")
        print("  1. Run: python colab_train.py")
        print("  2. After training, run: python colab_eval.py")
    else:
        print("⚠️ Some annotations failed. Check paths above.")
    print("=" * 60)

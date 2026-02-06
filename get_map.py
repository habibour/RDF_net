import os
import xml.etree.ElementTree as ET
import time
import json
from PIL import Image
from tqdm import tqdm
from utils.utils import get_classes
from utils.utils_map import get_map
from yolo import YOLO
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def read_file_with_retry(file_path, max_retries=10, delay=3):
    """Read a file with retry logic to handle I/O errors (common with Google Drive)."""
    for attempt in range(max_retries):
        try:
            return ET.parse(file_path).getroot()
        except (OSError, IOError) as e:
            if attempt < max_retries - 1:
                print(f"\nI/O error reading {file_path}, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(delay * (attempt + 1))  # Exponential backoff
            else:
                print(f"\nFailed to read {file_path} after {max_retries} attempts: {e}")
                return None
        except ET.ParseError as e:
            print(f"\nXML parse error in {file_path}: {e}")
            return None
    return None

def open_image_with_retry(image_path, max_retries=10, delay=3):
    """Open an image with retry logic to handle I/O errors."""
    for attempt in range(max_retries):
        try:
            img = Image.open(image_path)
            img.load()  # Force load the image data
            return img
        except (OSError, IOError) as e:
            if attempt < max_retries - 1:
                print(f"\nI/O error reading {image_path}, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(delay * (attempt + 1))  # Exponential backoff
            else:
                print(f"\nFailed to read {image_path} after {max_retries} attempts: {e}")
                return None
    return None

def save_checkpoint(checkpoint_path, phase, completed_ids):
    """Save progress checkpoint."""
    with open(checkpoint_path, 'w') as f:
        json.dump({'phase': phase, 'completed': list(completed_ids)}, f)

def load_checkpoint(checkpoint_path):
    """Load progress checkpoint."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
            return data.get('phase', 'predict'), set(data.get('completed', []))
    return 'predict', set()

if __name__ == "__main__":

    # ============================================================
    # REPRODUCE PAPER RESULTS - RDFNet Evaluation
    # Expected mAP results from paper:
    #   - RTTS:    59.93%
    #   - VOC-FOG: 78.39%
    #   - FDD:     36.99%
    # ============================================================
    
    dataname        = 'rtts'  # Options: 'rtts', 'fdd', 'VOC-FOG'
    classes_path    = 'model_data/rtts_classes.txt'
    model_path      = 'model_data/RDFNet.pth'
    MINOVERLAP      = 0.5
    confidence      = 0.001
    nms_iou         = 0.5
    score_threhold  = 0.5
    map_vis         = False
    
    # Resume from checkpoint if available
    resume_from_checkpoint = True

    if dataname == 'rtts':
        # Check for local machine paths first, then Colab paths
        possible_paths = [
            '/Users/habibourakash/Downloads/RTTS',  # Local Mac
            '/Users/habibourakash/CSERUET/4-2/CSE4200-theisis/object_detection/dataset/RTTS',
            './dataset/RTTS',  # Relative path
            '/content/RTTS',  # Colab local
            '/content/drive/MyDrive/dataset/RTTS',  # Colab Drive
        ]
        VOCdevkit_path = None
        for path in possible_paths:
            if os.path.exists(path):
                VOCdevkit_path = path
                break
        if VOCdevkit_path is None:
            print("ERROR: RTTS dataset not found! Please set VOCdevkit_path manually.")
            print("Searched paths:", possible_paths)
            exit(1)
        print("VOCdevkit_path:", VOCdevkit_path)
        print("Full test.txt path:", os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt"))
        print("Exists:", os.path.exists(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")))
    elif dataname == 'fdd': 
        # Use local path (copy from Drive first to avoid I/O errors)
        VOCdevkit_path  = '/content/FDD'
        if not os.path.exists(VOCdevkit_path):
            VOCdevkit_path = '/content/drive/MyDrive/dataset/FDD'
        print("VOCdevkit_path:", VOCdevkit_path)
        print("Full test.txt path:", os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt"))
        print("Exists:", os.path.exists(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")))
    elif dataname == 'VOC-FOG':
        # Use local path (copy from Drive first to avoid I/O errors)
        VOCdevkit_path  = '/content/VOC-FOG'
        if not os.path.exists(VOCdevkit_path):
            VOCdevkit_path = '/content/drive/MyDrive/dataset/VOC-FOG'
        print("VOCdevkit_path:", VOCdevkit_path)
        print("Full test.txt path:", os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt"))
        print("Exists:", os.path.exists(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")))

    map_out_path    = f'map_out-{dataname}'
    checkpoint_path = f'{map_out_path}/checkpoint.json'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))
    class_names, _ = get_classes(classes_path)

    # Load checkpoint if resuming
    current_phase, completed_ids = ('predict', set())
    if resume_from_checkpoint:
        current_phase, completed_ids = load_checkpoint(checkpoint_path)
        if completed_ids:
            print(f"Resuming from checkpoint: phase='{current_phase}', completed={len(completed_ids)} items")

    # ============================================================
    # PHASE 1: Get Predictions
    # ============================================================
    if current_phase == 'predict':
        print("Load model.")
        yolo = YOLO(model_path = model_path, confidence = confidence, nms_iou = nms_iou)
        print("Load model done.")
        print("Get predict result.")
        
        skipped_images = []
        # Filter out already completed images
        remaining_ids = [img_id for img_id in image_ids if os.path.splitext(img_id)[0] not in completed_ids]
        
        if len(remaining_ids) < len(image_ids):
            print(f"Skipping {len(image_ids) - len(remaining_ids)} already processed images...")
        
        for image_id in tqdm(remaining_ids, desc="Predictions"):
            base_id = os.path.splitext(image_id)[0]
            
            if dataname == 'VOC-FOG':
                if image_id.lower().endswith(('.jpg', '.png')):
                    image_path = os.path.join(VOCdevkit_path, "VOC2007/FOG/" + image_id)
                else:
                    image_path = os.path.join(VOCdevkit_path, "VOC2007/FOG/" + image_id + ".jpg")
            else:
                if image_id.lower().endswith(('.jpg', '.png')):
                    image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id)
                else:
                    image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/" + image_id + ".jpg")
            
            image = open_image_with_retry(image_path)
            if image is None:
                skipped_images.append(image_id)
                # Create empty detection file
                det_file = os.path.join(map_out_path, "detection-results", base_id + ".txt")
                open(det_file, "w").close()
                completed_ids.add(base_id)
                save_checkpoint(checkpoint_path, 'predict', completed_ids)
                continue
                
            if map_vis:
                ext = os.path.splitext(image_id)[1] or '.jpg'
                image.save(os.path.join(map_out_path, "images-optional/" + base_id + ext))
            
            yolo.get_map_txt(base_id, image, class_names, map_out_path)
            completed_ids.add(base_id)
            
            # Save checkpoint every 100 images
            if len(completed_ids) % 100 == 0:
                save_checkpoint(checkpoint_path, 'predict', completed_ids)
        
        if skipped_images:
            print(f"\nWarning: Skipped {len(skipped_images)} images due to I/O errors")
        print("Get predict result done.")
        
        # Move to ground truth phase
        current_phase = 'ground_truth'
        completed_ids = set()
        save_checkpoint(checkpoint_path, current_phase, completed_ids)

    # ============================================================
    # PHASE 2: Get Ground Truth
    # ============================================================
    if current_phase == 'ground_truth':
        print("Get ground truth result.")
        
        # Ensure detection-results files exist for every image
        detection_results_dir = os.path.join(map_out_path, "detection-results")
        for image_id in image_ids:
            base_id = os.path.splitext(image_id)[0]
            det_file = os.path.join(detection_results_dir, base_id + ".txt")
            if not os.path.exists(det_file):
                open(det_file, "w").close()

        skipped_annotations = []
        # Filter out already completed annotations
        remaining_ids = [img_id for img_id in image_ids if os.path.splitext(img_id)[0] not in completed_ids]
        
        if len(remaining_ids) < len(image_ids):
            print(f"Skipping {len(image_ids) - len(remaining_ids)} already processed annotations...")
        
        for image_id in tqdm(remaining_ids, desc="Ground Truth"):
            base_id = os.path.splitext(image_id)[0]
            annotation_path = os.path.join(VOCdevkit_path, "VOC2007/Annotations/" + base_id + ".xml")
            gt_file_path = os.path.join(map_out_path, "ground-truth/" + base_id + ".txt")
            
            # Use retry logic to read XML annotation
            root = read_file_with_retry(annotation_path)
            if root is None:
                skipped_annotations.append(base_id)
                # Create empty ground-truth file
                open(gt_file_path, "w").close()
                completed_ids.add(base_id)
                save_checkpoint(checkpoint_path, 'ground_truth', completed_ids)
                continue
            
            with open(gt_file_path, "w") as new_f:
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult') is not None:
                        difficult = obj.find('difficult').text
                        if int(difficult) == 1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text
                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
            
            completed_ids.add(base_id)
            
            # Save checkpoint every 100 annotations
            if len(completed_ids) % 100 == 0:
                save_checkpoint(checkpoint_path, 'ground_truth', completed_ids)
        
        if skipped_annotations:
            print(f"\nWarning: Skipped {len(skipped_annotations)} annotations due to I/O errors")
        print("Get ground truth result done.")
        
        # Move to map calculation phase
        current_phase = 'calculate_map'
        save_checkpoint(checkpoint_path, current_phase, set())

    # ============================================================
    # PHASE 3: Calculate mAP
    # ============================================================
    if current_phase == 'calculate_map':
        print("Get map.")
        get_map(MINOVERLAP, True, score_threhold=score_threhold, path=map_out_path)
        print("Get map done.")
        
        # Clean up checkpoint file after successful completion
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print("="*60)


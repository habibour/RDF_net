from random import sample, shuffle
import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import cvtColor, preprocess_input

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _find_image_path(images_dir, image_id):
    for ext in _IMAGE_EXTS:
        image_path = os.path.join(images_dir, image_id + ext)
        if os.path.exists(image_path):
            return image_path
    return None


def _resolve_clean_images_dir(clean_root, image_dir):
    if clean_root is None:
        return None
    candidate = os.path.join(clean_root, image_dir)
    if os.path.isdir(candidate):
        return candidate
    return clean_root


def build_voc2007_paired_annotation_lines(fog_root, clean_root, list_path, class_names, debug=False):
    """
    Build annotation lines for paired fog/clean VOC2007 dataset.
    
    Args:
        fog_root: Path to fog dataset root (e.g., VOC2007_FOG)
        clean_root: Path to clean dataset root (e.g., VOC2007_CLEAN) 
        list_path: Path to image list file (e.g., ImageSets/Main/trainval.txt)
        class_names: List of class names
        debug: Print debug info for first 3 samples
    
    Returns:
        fog_lines, clean_lines: Lists of annotation strings for fog and clean images
    """
    print(f"📍 Building paired annotation lines...")
    print(f"   Fog root: {fog_root}")
    print(f"   Clean root: {clean_root}")
    print(f"   List path: {list_path}")
    
    # Read image IDs from list file
    with open(list_path, 'r') as f:
        image_ids = [line.strip() for line in f.readlines() if line.strip()]
    
    fog_lines = []
    clean_lines = []
    errors = 0
    
    for i, image_id in enumerate(image_ids):
        # Find fog and clean image paths
        fog_img_dir = os.path.join(fog_root, "JPEGImages")
        clean_img_dir = os.path.join(clean_root, "JPEGImages")
        
        fog_img_path = _find_image_path(fog_img_dir, image_id)
        clean_img_path = _find_image_path(clean_img_dir, image_id)
        
        # Find annotation (same for both fog and clean)
        ann_path = os.path.join(fog_root, "Annotations", f"{image_id}.xml")
        if not os.path.exists(ann_path):
            ann_path = os.path.join(clean_root, "Annotations", f"{image_id}.xml")
        
        # Debug info for first 3 samples
        if debug and i < 3:
            print(f"\n🔍 Sample {i+1}: {image_id}")
            print(f"   Fog image: {fog_img_path} {'✅' if fog_img_path and os.path.exists(fog_img_path) else '❌'}")
            print(f"   Clean image: {clean_img_path} {'✅' if clean_img_path and os.path.exists(clean_img_path) else '❌'}")
            print(f"   Annotation: {ann_path} {'✅' if os.path.exists(ann_path) else '❌'}")
        
        # Check if all required files exist
        if not fog_img_path or not clean_img_path or not os.path.exists(ann_path):
            if i < 5:  # Show first few errors
                print(f"❌ Missing files for {image_id}")
            errors += 1
            continue
        
        # Parse XML annotation  
        try:
            tree = ET.parse(ann_path)
            root = tree.getroot()
            
            fog_boxes = []
            clean_boxes = []
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in class_names:
                    continue
                    
                class_id = class_names.index(class_name)
                bbox = obj.find('bndbox')
                
                xmin = int(float(bbox.find('xmin').text))
                ymin = int(float(bbox.find('ymin').text))
                xmax = int(float(bbox.find('xmax').text))
                ymax = int(float(bbox.find('ymax').text))
                
                box_str = f"{xmin},{ymin},{xmax},{ymax},{class_id}"
                fog_boxes.append(box_str)
                clean_boxes.append(box_str)
            
            if fog_boxes:  # Only add if there are valid objects
                fog_line = f"{fog_img_path} " + " ".join(fog_boxes)
                clean_line = f"{clean_img_path} " + " ".join(clean_boxes)
                
                fog_lines.append(fog_line)
                clean_lines.append(clean_line)
                
                if debug and i < 3:
                    print(f"   📦 Objects: {len(fog_boxes)}")
        
        except Exception as e:
            if i < 5:
                print(f"❌ Error parsing {ann_path}: {e}")
            errors += 1
            continue
    
    print(f"✅ Successfully processed: {len(fog_lines)} paired samples")
    if errors > 0:
        print(f"❌ Errors/missing: {errors} samples")
    
    return fog_lines, clean_lines


def build_voc_annotation_lines(list_path, dataset_root, class_names, image_dir="JPEGImages", ann_dir="Annotations", clean_root=None):
    if class_names is None:
        raise ValueError("class_names must be provided to build VOC annotation lines.")
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"List file not found: {list_path}")
    images_dir = os.path.join(dataset_root, image_dir)
    annotations_dir = os.path.join(dataset_root, ann_dir)
    clean_images_dir = _resolve_clean_images_dir(clean_root, image_dir) or images_dir

    with open(list_path, "r", encoding="utf-8") as f:
        image_ids = [line.strip() for line in f.readlines() if line.strip()]

    annotation_lines = []
    clean_lines = []
    kept_ids = []

    for image_id in image_ids:
        xml_path = os.path.join(annotations_dir, image_id + ".xml")
        image_path = _find_image_path(images_dir, image_id)
        clean_image_path = _find_image_path(clean_images_dir, image_id)

        if not image_path or not clean_image_path or not os.path.exists(xml_path):
            continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception:
            continue

        boxes = []
        for obj in root.iter("object"):
            cls_name = obj.find("name").text
            if cls_name not in class_names:
                continue
            cls_id = class_names.index(cls_name)
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            box = [
                int(float(bbox.find("xmin").text)),
                int(float(bbox.find("ymin").text)),
                int(float(bbox.find("xmax").text)),
                int(float(bbox.find("ymax").text)),
                cls_id,
            ]
            boxes.append(",".join(map(str, box)))

        if not boxes:
            continue

        line = image_path + " " + " ".join(boxes)
        clean_line = clean_image_path + " " + " ".join(boxes)
        annotation_lines.append(line)
        clean_lines.append(clean_line)
        kept_ids.append(image_id)

    return annotation_lines, clean_lines, kept_ids

class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train=True, anchors=None, anchors_mask=None, epoch_length=None):
        """
        YOLO Dataset for paired fog/clean images.
        
        Args:
            annotation_lines: List of strings in format "fog_path,clean_path xmin,ymin,xmax,ymax,class_id ..."
            input_shape: [height, width] 
            num_classes: Number of classes
            train: Training mode flag
            anchors: YOLO anchors (legacy, not used)
            anchors_mask: Anchor mask (legacy, not used)  
            epoch_length: Epoch length (legacy, not used)
        """
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.length = len(self.annotation_lines)
        
        # Debug first few samples
        if len(annotation_lines) > 0:
            print(f"📊 Dataset initialized with {self.length} samples")
            for i in range(min(3, len(annotation_lines))):
                line_parts = annotation_lines[i].split()
                paths = line_parts[0]
                fog_path, clean_path = paths.split(',')
                print(f"   Sample {i+1}: fog={fog_path}, clean={clean_path}")
                print(f"   Targets: {len(line_parts) - 1} boxes")

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        """
        Get dataset item with fog image, targets, and clean image.
        Returns: (fog_tensor, targets_tensor, clean_tensor)
        """
        index = index % self.length
        
        # Parse annotation line: "fog_path,clean_path xmin,ymin,xmax,ymax,class_id ..."
        line_parts = self.annotation_lines[index].split()
        paths = line_parts[0]
        fog_path, clean_path = paths.split(',')
        
        # Load and preprocess images
        fog_image, boxes, clean_image = self.get_random_data(fog_path, clean_path, line_parts[1:], 
                                                           self.input_shape, random=self.train)
        
        # Convert to tensors
        fog_tensor = np.transpose(preprocess_input(np.array(fog_image, dtype=np.float32)), (2, 0, 1))
        clean_tensor = np.transpose(preprocess_input(np.array(clean_image, dtype=np.float32)), (2, 0, 1))
        
        # Process bounding boxes
        boxes = np.array(boxes, dtype=np.float32)
        nL = len(boxes)
        labels_out = np.zeros((nL, 6))
        
        if nL:
            # Normalize coordinates
            boxes[:, [0, 2]] = boxes[:, [0, 2]] / self.input_shape[1]
            boxes[:, [1, 3]] = boxes[:, [1, 3]] / self.input_shape[0]
            boxes[:, 2:4] = boxes[:, 2:4] - boxes[:, 0:2]  # Convert to width/height
            boxes[:, 0:2] = boxes[:, 0:2] + boxes[:, 2:4] / 2  # Convert to center
            labels_out[:, 1] = boxes[:, -1]  # Class labels
            labels_out[:, 2:] = boxes[:, :4]  # Coordinates
        
        return fog_tensor, labels_out, clean_tensor
    
    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a
    
    def get_random_data(self, fog_path, clean_path, box_strings, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        """
        Load and preprocess paired fog/clean images with data augmentation.
        
        Args:
            fog_path: Path to foggy image
            clean_path: Path to clean image  
            box_strings: List of bounding box strings "xmin,ymin,xmax,ymax,class_id"
            input_shape: Target image shape [height, width]
            random: Enable data augmentation
        
        Returns:
            fog_image, boxes, clean_image
        """
        # Load images
        fog_image = Image.open(fog_path)
        fog_image = cvtColor(fog_image)
        clean_image = Image.open(clean_path)
        clean_image = cvtColor(clean_image)
        
        # Parse bounding boxes
        boxes = []
        for box_str in box_strings:
            if box_str.strip():
                coords = box_str.split(',')
                if len(coords) == 5:
                    xmin, ymin, xmax, ymax, class_id = map(float, coords)
                    boxes.append([xmin, ymin, xmax, ymax, class_id])
        
        boxes = np.array(boxes, dtype=np.float32)
        
        # Get image dimensions
        iw, ih = fog_image.size
        h, w = input_shape
        
        if not random:
            # Resize images without augmentation for validation
            scale = min(w/iw, h/ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2
            
            fog_image = fog_image.resize((nw, nh), Image.BICUBIC)
            clean_image = clean_image.resize((nw, nh), Image.BICUBIC)
            
            new_fog = Image.new('RGB', (w, h), (128, 128, 128))
            new_clean = Image.new('RGB', (w, h), (128, 128, 128))
            new_fog.paste(fog_image, (dx, dy))
            new_clean.paste(clean_image, (dx, dy))
            
            # Adjust boxes
            if len(boxes) > 0:
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + dx
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + dy
                boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
                boxes[:, 2][boxes[:, 2] > w] = w
                boxes[:, 3][boxes[:, 3] > h] = h
                box_w = boxes[:, 2] - boxes[:, 0]
                box_h = boxes[:, 3] - boxes[:, 1]
                boxes = boxes[np.logical_and(box_w > 1, box_h > 1)]
            
            return new_fog, boxes, new_clean
        
        # Random augmentation for training
        new_ar = iw / ih * self.rand(1-jitter, 1+jitter) / self.rand(1-jitter, 1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
            
        fog_image = fog_image.resize((nw, nh), Image.BICUBIC)
        clean_image = clean_image.resize((nw, nh), Image.BICUBIC)
        
        # Random placement
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        
        new_fog = Image.new('RGB', (w, h), (128, 128, 128))
        new_clean = Image.new('RGB', (w, h), (128, 128, 128))
        new_fog.paste(fog_image, (dx, dy))
        new_clean.paste(clean_image, (dx, dy))
        
        # Apply color augmentations to fog image only (keep clean image unchanged for consistency)
        if self.rand() < .5:
            new_fog = new_fog.transpose(Image.FLIP_LEFT_RIGHT)
        
        hue_shift = self.rand(-hue, hue)
        sat_shift = self.rand(1, sat) if self.rand() < .5 else 1/self.rand(1, sat)
        val_shift = self.rand(1, val) if self.rand() < .5 else 1/self.rand(1, val)
        
        x = cv2.cvtColor(np.array(new_fog, np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue_shift*360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat_shift
        x[..., 2] *= val_shift
        x[x[:,:, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        new_fog = Image.fromarray((cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255).astype(np.uint8))
        
        # Adjust boxes
        if len(boxes) > 0:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * nw / iw + dx
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * nh / ih + dy
            boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
            boxes[:, 2][boxes[:, 2] > w] = w
            boxes[:, 3][boxes[:, 3] > h] = h
            box_w = boxes[:, 2] - boxes[:, 0]
            box_h = boxes[:, 3] - boxes[:, 1]
            boxes = boxes[np.logical_and(box_w > 1, box_h > 1)]
        
        return new_fog, boxes, new_clean


def yolo_dataset_collate(batch):
    """
    Collate function for DataLoader.
    Handles batches of (fog_tensor, targets_tensor, clean_tensor).
    """
    images = []
    bboxes = []
    clear_images = []
    
    for i, (img, box, clear) in enumerate(batch):
        images.append(img)
        if len(box) > 0:
            box[:, 0] = i  # Set batch index
        bboxes.append(box)
        clear_images.append(clear)
    
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    if len(bboxes) > 0:
        bboxes = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    else:
        bboxes = torch.empty(0, 6).type(torch.FloatTensor)
    clear_images = torch.from_numpy(np.array(clear_images)).type(torch.FloatTensor)
    
    return images, bboxes, clear_images

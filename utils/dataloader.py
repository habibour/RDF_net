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
    def __init__(self, annotation_lines, clean_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length, train,
                 list_path=None, dataset_root=None, class_names=None, clean_root=None):
        super(YoloDataset, self).__init__()
        if list_path and dataset_root:
            self.annotation_lines, self.clean_lines, _ = build_voc_annotation_lines(
                list_path=list_path,
                dataset_root=dataset_root,
                class_names=class_names,
                clean_root=clean_root,
            )
        else:
            if annotation_lines is None:
                raise ValueError("annotation_lines cannot be None when list_path is not provided.")
            self.annotation_lines = annotation_lines
            self.clean_lines = clean_lines if clean_lines is not None else annotation_lines

        if len(self.annotation_lines) != len(self.clean_lines):
            raise ValueError("annotation_lines and clean_lines must be the same length.")
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.anchors            = anchors
        self.anchors_mask       = anchors_mask
        self.epoch_length       = epoch_length
        self.train              = train
        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)
        self.bbox_attrs         = 5 + num_classes

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        index  = index % self.length
        image, box, clearimg= self.get_random_data(self.annotation_lines[index],self.clean_lines[index], self.input_shape, random = self.train)
        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box         = np.array(box, dtype=np.float32)
        clearimg    = np.transpose(preprocess_input(np.array(clearimg, dtype=np.float32)), (2, 0, 1))
        nL          = len(box)
        labels_out  = np.zeros((nL, 6))
        if nL:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
            labels_out[:, 1] = box[:, -1]
            labels_out[:, 2:] = box[:, :4]
        return image, labels_out, clearimg
    
    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a
    
    def get_random_data(self, annotation_line,clean_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()
        clearline = clean_line.split()
        image   = Image.open(line[0])
        image   = cvtColor(image)
        clearimg = Image.open(clearline[0])
        clearimg = cvtColor(clearimg)
        iw, ih  = image.size
        h, w    = input_shape
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)
            clearimg = clearimg.resize((nw, nh), Image.BICUBIC)
            new_clearimg = Image.new('RGB', (w, h), (128, 128, 128))
            new_clearimg.paste(clearimg, (dx, dy))
            clear_image_data = np.array(new_clearimg, np.float32)
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
            return image_data, box, clear_image_data
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        clearimg = clearimg.resize((nw, nh), Image.BICUBIC)
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image
        new_clearimg = Image.new('RGB', (w, h), (128, 128, 128))
        new_clearimg.paste(clearimg, (dx, dy))
        clearimg = new_clearimg
        flip = self.rand()<.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            clearimg = clearimg.transpose(Image.FLIP_LEFT_RIGHT)
        image_data      = np.array(image, np.uint8)
        clear_image_data = np.array(clearimg, np.uint8)
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        hue1, sat1, val1 = cv2.split(cv2.cvtColor(clear_image_data, cv2.COLOR_RGB2HSV))
        dtype1 = clear_image_data.dtype
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        x1 = np.arange(0, 256, dtype=r.dtype)
        lut_hue1 = ((x1 * r[0]) % 180).astype(dtype)
        lut_sat1 = np.clip(x1 * r[1], 0, 255).astype(dtype)
        lut_val1 = np.clip(x1 * r[2], 0, 255).astype(dtype)
        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        clear_image_data = cv2.merge((cv2.LUT(hue1, lut_hue1), cv2.LUT(sat1, lut_sat1), cv2.LUT(val1, lut_val1)))
        clear_image_data = cv2.cvtColor(clear_image_data, cv2.COLOR_HSV2RGB)
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)]
        return image_data, box, clear_image_data
    
    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox
        
def yolo_dataset_collate(batch):
    images  = []
    bboxes  = []
    clearimg = []
    for i, (img, box, clear) in enumerate(batch):
        images.append(img)
        box[:, 0] = i
        bboxes.append(box)
        clearimg.append(clear)
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    clearimg = torch.from_numpy(np.array(clearimg)).type(torch.FloatTensor)
    return images, bboxes, clearimg

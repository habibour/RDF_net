#!/usr/bin/env python3
import argparse
import os
import random


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _collect_ids_with_images(images_dir):
    ids = set()
    if not os.path.isdir(images_dir):
        return ids
    for name in os.listdir(images_dir):
        base, ext = os.path.splitext(name)
        if ext.lower() in IMAGE_EXTS:
            ids.add(base)
    return ids


def _collect_ids_with_xml(ann_dir):
    ids = set()
    if not os.path.isdir(ann_dir):
        return ids
    for name in os.listdir(ann_dir):
        base, ext = os.path.splitext(name)
        if ext.lower() == ".xml":
            ids.add(base)
    return ids


def make_split(dataset_root, seed=42):
    """Create train/val/test splits for VOC_FOG_12K dataset"""
    
    print(f"🔍 Dataset root input: {dataset_root}")
    print(f"🔍 Current working directory: {os.getcwd()}")
    
    # Collect image IDs from both VOC2007 and VOC2012
    voc2007_fog = os.path.join(dataset_root, "VOC2007_FOG")
    voc2007_ann = os.path.join(dataset_root, "VOC2007_Annotations")
    voc2012_fog = os.path.join(dataset_root, "VOC2012_FOG")
    voc2012_ann = os.path.join(dataset_root, "VOC2012_Annotations")
    
    # Get valid IDs from both datasets
    voc2007_img_ids = _collect_ids_with_images(voc2007_fog)
    voc2007_ann_ids = _collect_ids_with_xml(voc2007_ann)
    voc2012_img_ids = _collect_ids_with_images(voc2012_fog)
    voc2012_ann_ids = _collect_ids_with_xml(voc2012_ann)
    
    # Only keep IDs that have both image and annotation
    valid_2007_ids = sorted(list(voc2007_img_ids & voc2007_ann_ids))
    valid_2012_ids = sorted(list(voc2012_img_ids & voc2012_ann_ids))
    
    print(f"VOC2007 valid samples: {len(valid_2007_ids)}")
    print(f"VOC2012 valid samples: {len(valid_2012_ids)}")
    
    # Combine all valid IDs
    all_valid_ids = valid_2007_ids + valid_2012_ids
    
    if not all_valid_ids:
        raise RuntimeError("No valid image ids found with both images and XML annotations.")
    
    # Create splits
    random.seed(seed)
    random.shuffle(all_valid_ids)

    n_total = len(all_valid_ids)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    train_ids = all_valid_ids[:n_train]
    val_ids = all_valid_ids[n_train:n_train + n_val]
    test_ids = all_valid_ids[n_train + n_val:]

    # FORCE write to writable location - never try to write to dataset_root in Kaggle
    splits_output_dir = "/kaggle/working/dataset_splits"
    imagesets_dir = os.path.join(splits_output_dir, "ImageSets", "Main")
    
    print(f"📁 Creating splits in: {imagesets_dir}")
    
    # Make sure directory exists
    os.makedirs(imagesets_dir, exist_ok=True)
    
    print(f"✅ Created directory: {imagesets_dir}")

    def _write_list(path, ids):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(ids))

    train_path = os.path.join(imagesets_dir, "train.txt")
    val_path = os.path.join(imagesets_dir, "val.txt")
    test_path = os.path.join(imagesets_dir, "test.txt")
    
    _write_list(train_path, train_ids)
    _write_list(val_path, val_ids)
    _write_list(test_path, test_ids)
    
    print(f"✅ Wrote splits to:")
    print(f"   📄 {train_path}")
    print(f"   📄 {val_path}")
    print(f"   📄 {test_path}")

    return n_train, n_val, n_test, n_total, splits_output_dir


def main():
    parser = argparse.ArgumentParser(description="Create 80/10/10 split files for VOC_FOG_12K_Upload.")
    parser.add_argument("--dataset_root", required=True, help="Path to VOC_FOG_12K_Upload directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    n_train, n_val, n_test, n_total, out_dir = make_split(args.dataset_root, seed=args.seed)
    print(f"Dataset root: {args.dataset_root}")
    print(f"ImageSets created in: {os.path.join(out_dir, 'ImageSets', 'Main')}")
    print(f"Total: {n_total} | Train: {n_train} | Val: {n_val} | Test: {n_test}")


if __name__ == "__main__":
    main()

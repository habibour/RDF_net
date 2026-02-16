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
    images_dir = os.path.join(dataset_root, "JPEGImages")
    ann_dir = os.path.join(dataset_root, "Annotations")
    imagesets_dir = os.path.join(dataset_root, "ImageSets", "Main")
    os.makedirs(imagesets_dir, exist_ok=True)

    image_ids = _collect_ids_with_images(images_dir)
    ann_ids = _collect_ids_with_xml(ann_dir)
    valid_ids = sorted(list(image_ids & ann_ids))

    if not valid_ids:
        raise RuntimeError("No valid image ids found with both images and XML annotations.")

    random.seed(seed)
    random.shuffle(valid_ids)

    n_total = len(valid_ids)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    train_ids = valid_ids[:n_train]
    val_ids = valid_ids[n_train:n_train + n_val]
    test_ids = valid_ids[n_train + n_val:]

    def _write_list(path, ids):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(ids))

    _write_list(os.path.join(imagesets_dir, "train.txt"), train_ids)
    _write_list(os.path.join(imagesets_dir, "val.txt"), val_ids)
    _write_list(os.path.join(imagesets_dir, "test.txt"), test_ids)

    return n_train, n_val, n_test, n_total, imagesets_dir


def main():
    parser = argparse.ArgumentParser(description="Create 80/10/10 split files for VOC_FOG_12K_Upload.")
    parser.add_argument("--dataset_root", required=True, help="Path to VOC_FOG_12K_Upload directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    n_train, n_val, n_test, n_total, out_dir = make_split(args.dataset_root, seed=args.seed)
    print(f"Dataset root: {args.dataset_root}")
    print(f"ImageSets path: {out_dir}")
    print(f"Total: {n_total} | Train: {n_train} | Val: {n_val} | Test: {n_test}")


if __name__ == "__main__":
    main()

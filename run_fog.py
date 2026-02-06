#!/usr/bin/env python3
"""Quick script to generate foggy images for training."""

import cv2
import numpy as np
import os
import random
from glob import glob

def add_fog(image, beta=1.0):
    """Add synthetic fog using Atmospheric Scattering Model."""
    height, width = image.shape[:2]
    img_float = image.astype(np.float32) / 255.0
    
    # Generate depth map (gradient from top to bottom)
    depth = np.linspace(0.1, 1.0, height).reshape(-1, 1)
    depth = np.tile(depth, (1, width))
    
    # Transmission map: t(x) = exp(-beta * d(x))
    transmission = np.exp(-beta * depth)
    transmission = np.clip(transmission, 0.1, 1.0)[:, :, np.newaxis]
    
    # Atmospheric light (slightly gray-white)
    A = np.array([0.9, 0.9, 0.9])
    
    # ASM: I(x) = J(x) * t(x) + A * (1 - t(x))
    foggy = img_float * transmission + A * (1 - transmission)
    return np.clip(foggy * 255, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    input_dir = "../VOC2007/JPEGImages"
    output_dir = "../VOC2007/FOG"
    
    os.makedirs(output_dir, exist_ok=True)
    
    files = sorted(glob(os.path.join(input_dir, "*.jpg")))
    total = len(files)
    
    print(f"Generating foggy images: {total} images")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        outpath = os.path.join(output_dir, fname)
        
        if os.path.exists(outpath):
            continue
        
        img = cv2.imread(fpath)
        if img is None:
            print(f"Warning: Could not read {fpath}")
            continue
        
        beta = random.uniform(0.6, 1.5)
        foggy = add_fog(img, beta=beta)
        cv2.imwrite(outpath, foggy)
        
        if (i + 1) % 500 == 0:
            print(f"Progress: {i+1}/{total} ({100*(i+1)/total:.1f}%)")
    
    print(f"Done! Generated {len(os.listdir(output_dir))} foggy images.")

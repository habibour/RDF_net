#!/usr/bin/env python3
"""
Generate synthetic fog dataset from VOC2007 using atmospheric scattering model.
Creates paired fog/clean dataset for RDFNet training.
"""

import os
import sys
import argparse
import shutil
import numpy as np
import cv2
from pathlib import Path
import random
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Generate VOC2007 synthetic fog dataset')
    parser.add_argument('--voc2007_root', type=str, required=True,
                        help='Path to VOCdevkit/VOC2007 directory')
    parser.add_argument('--out_root', type=str, required=True,
                        help='Output base directory (e.g., /data/VOC2007_SYNFOG)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--beta_min', type=float, default=0.6,
                        help='Minimum scattering coefficient')
    parser.add_argument('--beta_max', type=float, default=2.2,
                        help='Maximum scattering coefficient')
    parser.add_argument('--A_min', type=float, default=0.75,
                        help='Minimum atmospheric light')
    parser.add_argument('--A_max', type=float, default=1.0,
                        help='Maximum atmospheric light')
    return parser.parse_args()

def create_pseudo_depth_map(height, width, seed):
    """
    Create a pseudo-depth map using gradients and noise.
    Returns normalized depth in [0,1] where 0=near, 1=far.
    """
    np.random.seed(seed)
    
    # Create coordinate grids
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    
    # Combine horizontal and vertical gradients
    h_gradient = X  # Left to right depth increase
    v_gradient = Y  # Top to bottom depth increase
    
    # Add some randomness
    noise = np.random.rand(height, width) * 0.3
    
    # Combine gradients with different weights
    depth = 0.4 * h_gradient + 0.4 * v_gradient + 0.2 * noise
    
    # Apply Gaussian blur for smoothness
    depth = cv2.GaussianBlur(depth.astype(np.float32), (21, 21), 5.0)
    
    # Normalize to [0, 1]
    depth = (depth - depth.min()) / (depth.max() - depth.min())
    
    return depth

def apply_atmospheric_scattering(image, beta, A, depth_map):
    """
    Apply atmospheric scattering fog model: I = J*t + A*(1-t)
    where t = exp(-beta * depth)
    
    Args:
        image: Input clean image (H,W,3) in [0,255]
        beta: Scattering coefficient
        A: Atmospheric light [0,1]
        depth_map: Normalized depth map (H,W) in [0,1]
    
    Returns:
        Foggy image (H,W,3) in [0,255]
    """
    # Normalize image to [0,1]
    J = image.astype(np.float32) / 255.0
    
    # Calculate transmission map: t = exp(-beta * depth)
    t = np.exp(-beta * depth_map)
    t = np.expand_dims(t, axis=2)  # (H,W,1) for broadcasting
    
    # Apply atmospheric scattering model
    I = J * t + A * (1 - t)
    
    # Clamp to [0,1] and convert back to [0,255]
    I = np.clip(I, 0, 1)
    I = (I * 255).astype(np.uint8)
    
    return I

def copy_directory_structure(src_dir, dst_dir, file_patterns=None):
    """
    Copy directory structure and files matching patterns.
    
    Args:
        src_dir: Source directory
        dst_dir: Destination directory  
        file_patterns: List of file extensions to copy (e.g., ['.xml', '.txt'])
    """
    os.makedirs(dst_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(src_dir):
        # Create corresponding directory structure
        rel_path = os.path.relpath(root, src_dir)
        dst_root = os.path.join(dst_dir, rel_path) if rel_path != '.' else dst_dir
        os.makedirs(dst_root, exist_ok=True)
        
        # Copy files matching patterns
        for file in files:
            if file_patterns is None or any(file.endswith(ext) for ext in file_patterns):
                src_file = os.path.join(root, file)
                dst_file = os.path.join(dst_root, file)
                shutil.copy2(src_file, dst_file)

def main():
    args = parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Validate input directory
    voc2007_root = Path(args.voc2007_root)
    if not voc2007_root.exists():
        print(f"Error: VOC2007 root directory not found: {voc2007_root}")
        sys.exit(1)
    
    # Required subdirectories
    required_dirs = ['JPEGImages', 'Annotations', 'ImageSets']
    for req_dir in required_dirs:
        if not (voc2007_root / req_dir).exists():
            print(f"Error: Required directory not found: {voc2007_root / req_dir}")
            sys.exit(1)
    
    # Create output directory structure
    out_root = Path(args.out_root)
    clean_root = out_root / 'VOC2007_CLEAN'
    fog_root = out_root / 'VOC2007_FOG'
    
    print(f"🚀 Creating VOC2007 synthetic fog dataset...")
    print(f"📁 Input: {voc2007_root}")
    print(f"📁 Output: {out_root}")
    print(f"🌫️  Fog params: β=[{args.beta_min}, {args.beta_max}], A=[{args.A_min}, {args.A_max}]")
    
    # 1. Copy clean dataset (original VOC2007)
    print("\n📋 Step 1: Copying clean dataset...")
    os.makedirs(clean_root, exist_ok=True)
    
    # Copy Annotations
    print("   Copying Annotations...")
    shutil.copytree(voc2007_root / 'Annotations', clean_root / 'Annotations', dirs_exist_ok=True)
    
    # Copy ImageSets
    print("   Copying ImageSets...")
    shutil.copytree(voc2007_root / 'ImageSets', clean_root / 'ImageSets', dirs_exist_ok=True)
    
    # Copy or symlink JPEGImages
    print("   Copying JPEGImages...")
    shutil.copytree(voc2007_root / 'JPEGImages', clean_root / 'JPEGImages', dirs_exist_ok=True)
    
    # 2. Create fog dataset structure
    print("\n🌫️  Step 2: Creating fog dataset structure...")
    os.makedirs(fog_root, exist_ok=True)
    
    # Copy Annotations and ImageSets (same as clean)
    print("   Copying Annotations...")
    shutil.copytree(voc2007_root / 'Annotations', fog_root / 'Annotations', dirs_exist_ok=True)
    
    print("   Copying ImageSets...")
    shutil.copytree(voc2007_root / 'ImageSets', fog_root / 'ImageSets', dirs_exist_ok=True)
    
    # Create JPEGImages directory for fog images
    os.makedirs(fog_root / 'JPEGImages', exist_ok=True)
    
    # 3. Generate fog images
    print("\n🌫️  Step 3: Generating synthetic fog images...")
    
    images_dir = voc2007_root / 'JPEGImages'
    image_files = list(images_dir.glob('*.jpg'))
    
    if not image_files:
        print("Error: No JPEG images found in JPEGImages directory")
        sys.exit(1)
    
    print(f"   Processing {len(image_files)} images...")
    
    fog_params_log = []
    
    for img_file in tqdm(image_files, desc="Generating fog"):
        # Load clean image
        clean_img = cv2.imread(str(img_file))
        if clean_img is None:
            print(f"Warning: Could not load image {img_file}")
            continue
        
        height, width = clean_img.shape[:2]
        
        # Generate random fog parameters for this image
        beta = random.uniform(args.beta_min, args.beta_max)
        A = random.uniform(args.A_min, args.A_max)
        
        # Create pseudo-depth map
        img_seed = hash(img_file.stem) % (2**31)  # Deterministic seed per image
        depth_map = create_pseudo_depth_map(height, width, img_seed)
        
        # Apply atmospheric scattering
        fog_img = apply_atmospheric_scattering(clean_img, beta, A, depth_map)
        
        # Save fog image
        fog_img_path = fog_root / 'JPEGImages' / img_file.name
        cv2.imwrite(str(fog_img_path), fog_img)
        
        # Log parameters
        fog_params_log.append({
            'image': img_file.name,
            'beta': beta,
            'A': A,
            'seed': img_seed
        })
    
    # 4. Write dataset summary
    print("\n📊 Step 4: Writing dataset summary...")
    
    # Count files
    clean_images = len(list((clean_root / 'JPEGImages').glob('*.jpg')))
    fog_images = len(list((fog_root / 'JPEGImages').glob('*.jpg')))
    annotations = len(list((clean_root / 'Annotations').glob('*.xml')))
    
    # Write README
    readme_content = f"""# VOC2007 Synthetic Fog Dataset

Generated on: {np.datetime64('now')}

## Parameters Used
- Random seed: {args.seed}
- Scattering coefficient (β): [{args.beta_min}, {args.beta_max}]
- Atmospheric light (A): [{args.A_min}, {args.A_max}]

## Dataset Statistics
- Clean images: {clean_images}
- Fog images: {fog_images}
- Annotations: {annotations}

## Directory Structure
```
VOC2007_SYNFOG/
├── VOC2007_CLEAN/
│   ├── JPEGImages/      # Original clean images
│   ├── Annotations/     # XML annotation files
│   └── ImageSets/       # Train/val/test splits
└── VOC2007_FOG/
    ├── JPEGImages/      # Synthetic fog images
    ├── Annotations/     # Same XML files as clean
    └── ImageSets/       # Same splits as clean
```

## Fog Generation Model
Atmospheric Scattering: I = J*t + A*(1-t)
- I: Observed foggy image
- J: Scene radiance (clean image)
- t: Transmission map = exp(-β*d)
- A: Atmospheric light
- d: Pseudo-depth map (gradient + noise + blur)

## Usage
Use this dataset for training dehazing + object detection models like RDFNet.
The clean and fog images are paired by filename for supervised learning.
"""
    
    with open(out_root / 'README.md', 'w') as f:
        f.write(readme_content)
    
    # Save fog parameters log
    import json
    with open(out_root / 'fog_parameters.json', 'w') as f:
        json.dump(fog_params_log, f, indent=2)
    
    print(f"\n✅ Dataset generation complete!")
    print(f"📁 Output directory: {out_root}")
    print(f"📊 Clean images: {clean_images}")
    print(f"🌫️  Fog images: {fog_images}")
    print(f"📋 Annotations: {annotations}")
    print(f"📄 See README.md and fog_parameters.json for details")

if __name__ == '__main__':
    main()
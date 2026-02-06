"""
Generate foggy images using Atmospheric Scattering Model (ASM)
Formula: I(x) = J(x) * t(x) + A * (1 - t(x))
Where:
    I(x) = Foggy image
    J(x) = Clear image
    t(x) = Transmission map = exp(-beta * d(x))
    A = Atmospheric light (usually white: [1, 1, 1])
    beta = Scattering coefficient (controls fog density)
    d(x) = Depth map (estimated or random)
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import random

def generate_depth_map(height, width, mode='gradient'):
    """Generate a synthetic depth map."""
    if mode == 'gradient':
        # Linear gradient from top to bottom (sky to ground)
        depth = np.linspace(0.1, 1.0, height).reshape(-1, 1)
        depth = np.tile(depth, (1, width))
    elif mode == 'random':
        # Random depth with smoothing
        depth = np.random.rand(height // 8, width // 8)
        depth = cv2.resize(depth, (width, height))
        depth = cv2.GaussianBlur(depth, (21, 21), 0)
        depth = 0.1 + 0.9 * depth  # Scale to [0.1, 1.0]
    elif mode == 'center':
        # Radial depth from center
        y, x = np.ogrid[:height, :width]
        cy, cx = height // 2, width // 2
        depth = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        depth = depth / depth.max()
        depth = 0.1 + 0.9 * depth
    else:
        depth = np.ones((height, width)) * 0.5
    
    return depth.astype(np.float32)

def add_fog(image, beta=None, A=None, depth_mode='gradient'):
    """
    Add synthetic fog to an image using ASM.
    
    Args:
        image: Input clear image (BGR, uint8)
        beta: Scattering coefficient (0.5-2.0, higher = more fog)
        A: Atmospheric light [0-1] for each channel
        depth_mode: 'gradient', 'random', or 'center'
    
    Returns:
        Foggy image (BGR, uint8)
    """
    if beta is None:
        beta = random.uniform(0.6, 1.5)  # Random fog density
    
    if A is None:
        # Slightly vary atmospheric light for realism
        A = np.array([0.8 + random.uniform(0, 0.2)] * 3)
    
    height, width = image.shape[:2]
    
    # Normalize image to [0, 1]
    img_float = image.astype(np.float32) / 255.0
    
    # Generate depth map
    depth = generate_depth_map(height, width, mode=depth_mode)
    
    # Calculate transmission map: t(x) = exp(-beta * d(x))
    transmission = np.exp(-beta * depth)
    transmission = np.clip(transmission, 0.1, 1.0)  # Avoid complete fog
    transmission = transmission[:, :, np.newaxis]  # Expand dims for broadcasting
    
    # Apply ASM: I(x) = J(x) * t(x) + A * (1 - t(x))
    foggy = img_float * transmission + A * (1 - transmission)
    
    # Clip and convert back to uint8
    foggy = np.clip(foggy * 255, 0, 255).astype(np.uint8)
    
    return foggy

def process_dataset(input_dir, output_dir, beta_range=(0.6, 1.5)):
    """
    Process all images in a directory and add fog.
    
    Args:
        input_dir: Path to clear images (JPEGImages)
        output_dir: Path to save foggy images (FOG)
        beta_range: Range of fog density (min, max)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Processing {len(image_files)} images...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Beta range: {beta_range}")
    
    for img_file in tqdm(image_files, desc="Adding fog"):
        img_path = os.path.join(input_dir, img_file)
        out_path = os.path.join(output_dir, img_file)
        
        # Skip if already processed
        if os.path.exists(out_path):
            continue
        
        # Read image
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not read {img_path}")
            continue
        
        # Random fog parameters for variety
        beta = random.uniform(*beta_range)
        depth_mode = random.choice(['gradient', 'random', 'center'])
        
        # Add fog
        foggy_image = add_fog(image, beta=beta, depth_mode=depth_mode)
        
        # Save
        cv2.imwrite(out_path, foggy_image)
    
    print(f"Done! Foggy images saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate foggy images using ASM')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to clear images (JPEGImages folder)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save foggy images (FOG folder)')
    parser.add_argument('--beta_min', type=float, default=0.6,
                        help='Minimum fog density (default: 0.6)')
    parser.add_argument('--beta_max', type=float, default=1.5,
                        help='Maximum fog density (default: 1.5)')
    
    args = parser.parse_args()
    
    process_dataset(
        input_dir=args.input,
        output_dir=args.output,
        beta_range=(args.beta_min, args.beta_max)
    )

#!/usr/bin/env python3
"""
Fog Application Script for VOC2012_FOGGY Images
==============================================

Applies atmospheric scattering fog to clean images in VOC2012_FOGGY directory.
The images are currently clean and need fog synthesis for proper training.

Uses Koschmieder's law: I(x) = J(x) * t(x) + A * (1 - t(x))
Where:
- I(x) = observed foggy image
- J(x) = original clean image  
- t(x) = transmission map (visibility through fog)
- A = atmospheric light (fog color/brightness)
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import shutil
from typing import Tuple, Optional
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fog_application.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FogGenerator:
    """Atmospheric scattering fog generator using Koschmieder's law."""
    
    def __init__(self, 
                 beta_min: float = 0.08,
                 beta_max: float = 0.25,
                 atmospheric_light: float = 0.75):
        """
        Initialize fog generator parameters.
        
        Args:
            beta_min: Minimum scattering coefficient (light fog)
            beta_max: Maximum scattering coefficient (dense fog)  
            atmospheric_light: Atmospheric light intensity [0.0, 1.0]
        """
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.atmospheric_light = atmospheric_light
        
    def estimate_depth_map(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map using gradient and dark channel priors.
        
        Args:
            image: Input BGR image
            
        Returns:
            Normalized depth map [0, 1]
        """
        # Convert to grayscale for gradient computation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute gradients (edges typically indicate depth changes)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Dark channel prior (darker areas typically more distant)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        dark_channel = cv2.erode(np.min(image, axis=2), kernel)
        
        # Combine gradient and dark channel for depth estimation
        # Normalize both to [0, 1]
        gradient_norm = cv2.GaussianBlur(gradient_magnitude, (5, 5), 0)
        gradient_norm = (gradient_norm - gradient_norm.min()) / (gradient_norm.max() - gradient_norm.min() + 1e-8)
        
        dark_norm = (dark_channel - dark_channel.min()) / (dark_channel.max() - dark_channel.min() + 1e-8)
        dark_norm = 1.0 - dark_norm  # Invert: darker = more distant
        
        # Weighted combination for depth map
        depth_map = 0.7 * dark_norm + 0.3 * gradient_norm
        
        # Smooth depth transitions
        depth_map = cv2.GaussianBlur(depth_map, (9, 9), 0)
        
        # Add some spatial variation (distant areas typically at top/background)
        h, w = depth_map.shape
        y_gradient = np.linspace(0, 0.3, h).reshape(-1, 1)
        spatial_bias = np.tile(y_gradient, (1, w))
        depth_map = np.clip(depth_map + spatial_bias, 0, 1)
        
        return depth_map
    
    def generate_transmission_map(self, depth_map: np.ndarray, beta: float) -> np.ndarray:
        """
        Generate transmission map from depth using exponential decay.
        
        Args:
            depth_map: Normalized depth map [0, 1]
            beta: Scattering coefficient
            
        Returns:
            Transmission map [0, 1]
        """
        # Koschmieder's law: t(x) = e^(-β * d(x))
        # Scale depth to reasonable distance range (0-10 units)
        scaled_depth = depth_map * 10.0
        transmission = np.exp(-beta * scaled_depth)
        
        # Ensure minimum transmission to prevent complete opacity
        transmission = np.clip(transmission, 0.3, 1.0)
        
        return transmission
    
    def apply_fog(self, 
                  image: np.ndarray, 
                  fog_density: str = 'medium') -> Tuple[np.ndarray, dict]:
        """
        Apply atmospheric scattering fog to image.
        
        Args:
            image: Input BGR image [0, 255]
            fog_density: 'light', 'medium', or 'dense'
            
        Returns:
            Tuple of (foggy_image, metadata)
        """
        # Fog density parameters
        density_params = {
            'light': {'beta_range': (self.beta_min, self.beta_min + 0.05), 'atm_light': 0.7},
            'medium': {'beta_range': (self.beta_min + 0.05, self.beta_max - 0.05), 'atm_light': 0.75},
            'dense': {'beta_range': (self.beta_max - 0.05, self.beta_max), 'atm_light': 0.8}
        }
        
        params = density_params.get(fog_density, density_params['medium'])
        
        # Randomly sample fog parameters
        beta = np.random.uniform(params['beta_range'][0], params['beta_range'][1])
        atm_light = params['atm_light'] + np.random.uniform(-0.02, 0.02)
        atm_light = np.clip(atm_light, 0.65, 0.85)
        
        # Normalize image to [0, 1]
        image_norm = image.astype(np.float32) / 255.0
        
        # Generate depth and transmission maps
        depth_map = self.estimate_depth_map(image)
        transmission = self.generate_transmission_map(depth_map, beta)
        
        # Expand transmission to 3 channels
        transmission_3ch = np.expand_dims(transmission, axis=2)
        transmission_3ch = np.repeat(transmission_3ch, 3, axis=2)
        
        # Apply Koschmieder's law: I = J * t + A * (1 - t)
        foggy_image_norm = (image_norm * transmission_3ch + 
                           atm_light * (1 - transmission_3ch))
        
        # Add slight color tint (fog typically has bluish/grayish tint)
        color_tint = np.array([0.99, 1.0, 1.01])  # Very subtle blue tint
        foggy_image_norm *= color_tint
        
        # Convert back to [0, 255] and ensure valid range
        foggy_image = np.clip(foggy_image_norm * 255.0, 0, 255).astype(np.uint8)
        
        # Metadata for debugging/validation
        metadata = {
            'fog_density': fog_density,
            'beta': float(beta),
            'atmospheric_light': float(atm_light),
            'avg_transmission': float(np.mean(transmission)),
            'min_transmission': float(np.min(transmission)),
            'max_transmission': float(np.max(transmission))
        }
        
        return foggy_image, metadata

def validate_paths(clean_dir: str, foggy_dir: str, backup_dir: Optional[str] = None) -> bool:
    """Validate input/output directory paths."""
    if not os.path.exists(clean_dir):
        logger.error(f"Clean images directory not found: {clean_dir}")
        return False
    
    if not os.path.exists(foggy_dir):
        logger.error(f"Foggy images directory not found: {foggy_dir}")
        return False
    
    # Check if foggy directory has images
    foggy_images = [f for f in os.listdir(foggy_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not foggy_images:
        logger.error(f"No images found in foggy directory: {foggy_dir}")
        return False
    
    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        logger.info(f"Backup directory ready: {backup_dir}")
    
    return True

def backup_original_images(foggy_dir: str, backup_dir: str) -> int:
    """
    Backup original clean images before fog application.
    
    Returns:
        Number of images backed up
    """
    logger.info("Backing up original clean images...")
    
    image_files = [f for f in os.listdir(foggy_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    backup_count = 0
    for img_file in tqdm(image_files, desc="Backing up"):
        src_path = os.path.join(foggy_dir, img_file)
        dst_path = os.path.join(backup_dir, img_file)
        
        if not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
            backup_count += 1
    
    logger.info(f"Backed up {backup_count} images to {backup_dir}")
    return backup_count

def apply_fog_to_dataset(clean_dir: str, 
                        foggy_dir: str,
                        fog_generator: FogGenerator,
                        backup_dir: Optional[str] = None,
                        fog_density: str = 'medium',
                        sample_limit: Optional[int] = None) -> dict:
    """
    Apply fog to all images in the dataset.
    
    Args:
        clean_dir: Directory containing original clean images
        foggy_dir: Directory containing images to convert to foggy
        fog_generator: FogGenerator instance
        backup_dir: Optional backup directory for original images
        fog_density: Fog density level
        sample_limit: Limit number of images to process (for testing)
        
    Returns:
        Processing statistics dictionary
    """
    # Get list of images to process
    image_files = [f for f in os.listdir(foggy_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if sample_limit:
        image_files = image_files[:sample_limit]
        logger.info(f"Processing limited sample: {len(image_files)} images")
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Backup original images if requested
    if backup_dir:
        backup_original_images(foggy_dir, backup_dir)
    
    # Process images
    stats = {
        'total_images': len(image_files),
        'processed': 0,
        'skipped': 0,
        'errors': 0,
        'metadata': []
    }
    
    for img_file in tqdm(image_files, desc=f"Applying {fog_density} fog"):
        try:
            img_path = os.path.join(foggy_dir, img_file)
            
            # Read original image
            original_img = cv2.imread(img_path)
            if original_img is None:
                logger.warning(f"Could not read image: {img_file}")
                stats['skipped'] += 1
                continue
            
            # Apply fog
            foggy_img, metadata = fog_generator.apply_fog(original_img, fog_density)
            
            # Save foggy image (overwrite original)
            success = cv2.imwrite(img_path, foggy_img)
            if not success:
                logger.error(f"Failed to save foggy image: {img_file}")
                stats['errors'] += 1
                continue
            
            # Store metadata
            metadata['filename'] = img_file
            stats['metadata'].append(metadata)
            stats['processed'] += 1
            
        except Exception as e:
            logger.error(f"Error processing {img_file}: {str(e)}")
            stats['errors'] += 1
    
    return stats

def main():
    """Main fog application script."""
    parser = argparse.ArgumentParser(description='Apply fog to VOC2012_FOGGY images')
    parser.add_argument('--clean-dir', 
                       default='../VOC_FOG_12K_Upload/VOC2012_FOG',
                       help='Directory with original clean images')
    parser.add_argument('--foggy-dir',
                       default='../VOC_FOG_12K_Upload/VOC2012_FOGGY', 
                       help='Directory with images to make foggy')
    parser.add_argument('--backup-dir',
                       default='../VOC_FOG_12K_Upload/VOC2012_CLEAN_BACKUP',
                       help='Backup directory for original images')
    parser.add_argument('--fog-density',
                       choices=['light', 'medium', 'dense'],
                       default='medium',
                       help='Fog density level')
    parser.add_argument('--sample-limit',
                       type=int,
                       help='Limit number of images to process (for testing)')
    parser.add_argument('--no-backup',
                       action='store_true',
                       help='Skip backing up original images')
    
    args = parser.parse_args()
    
    # Setup paths
    clean_dir = os.path.abspath(args.clean_dir)
    foggy_dir = os.path.abspath(args.foggy_dir)
    backup_dir = None if args.no_backup else os.path.abspath(args.backup_dir)
    
    logger.info("=== Fog Application Started ===")
    logger.info(f"Clean images directory: {clean_dir}")
    logger.info(f"Foggy images directory: {foggy_dir}")
    logger.info(f"Backup directory: {backup_dir}")
    logger.info(f"Fog density: {args.fog_density}")
    
    # Validate paths
    if not validate_paths(clean_dir, foggy_dir, backup_dir):
        logger.error("Path validation failed. Exiting.")
        return 1
    
    # Initialize fog generator
    fog_generator = FogGenerator()
    
    # Apply fog to dataset
    try:
        stats = apply_fog_to_dataset(
            clean_dir=clean_dir,
            foggy_dir=foggy_dir,
            fog_generator=fog_generator,
            backup_dir=backup_dir,
            fog_density=args.fog_density,
            sample_limit=args.sample_limit
        )
        
        # Print results
        logger.info("=== Processing Complete ===")
        logger.info(f"Total images: {stats['total_images']}")
        logger.info(f"Successfully processed: {stats['processed']}")
        logger.info(f"Skipped: {stats['skipped']}")
        logger.info(f"Errors: {stats['errors']}")
        
        # Save metadata
        if stats['metadata']:
            metadata_file = 'fog_application_metadata.json'
            with open(metadata_file, 'w') as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Metadata saved to: {metadata_file}")
        
        # Calculate average fog parameters
        if stats['metadata']:
            avg_beta = np.mean([m['beta'] for m in stats['metadata']])
            avg_atm_light = np.mean([m['atmospheric_light'] for m in stats['metadata']])
            avg_transmission = np.mean([m['avg_transmission'] for m in stats['metadata']])
            
            logger.info(f"Average fog parameters:")
            logger.info(f"  Beta (scattering): {avg_beta:.3f}")
            logger.info(f"  Atmospheric light: {avg_atm_light:.3f}")
            logger.info(f"  Transmission: {avg_transmission:.3f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error during processing: {str(e)}")
        return 1

if __name__ == '__main__':
    exit(main())
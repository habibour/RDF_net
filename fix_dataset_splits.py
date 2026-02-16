#!/usr/bin/env python3
"""
Debug and fix dataset splits path issue for Kaggle training.
This script helps diagnose the path issue and provides a working solution.
"""

import os
import shutil

def debug_paths():
    """Debug the dataset splits path issue"""
    print("🔍 Debugging dataset splits paths...")
    
    # Check all possible locations
    locations = [
        "/kaggle/working/dataset_splits/ImageSets/Main",
        "/kaggle/working/VOC_FOG_12K_Upload/ImageSets/Main", 
        "/kaggle/input/*/VOC_FOG_12K_Upload/ImageSets/Main"
    ]
    
    for loc in locations:
        print(f"\n📍 Checking: {loc}")
        if os.path.exists(loc):
            print(f"   ✅ Directory exists")
            split_files = ["train.txt", "val.txt", "test.txt"]
            for sf in split_files:
                fp = os.path.join(loc, sf)
                if os.path.exists(fp):
                    with open(fp, 'r') as f:
                        count = len(f.readlines())
                    print(f"   📄 {sf}: {count} samples")
                else:
                    print(f"   ❌ Missing: {sf}")
        else:
            print(f"   ❌ Directory not found")
    
    # Check symlink status
    voc_path = "/kaggle/working/VOC_FOG_12K_Upload"
    print(f"\n🔗 Checking symlink: {voc_path}")
    if os.path.exists(voc_path):
        if os.path.islink(voc_path):
            target = os.readlink(voc_path)
            print(f"   ✅ Symlink points to: {target}")
            print(f"   📁 Target writable: {os.access(target, os.W_OK)}")
        else:
            print(f"   📁 Regular directory")
    else:
        print(f"   ❌ Path not found")

def fix_splits_path():
    """Fix the dataset splits path issue"""
    print("\n🛠️  Fixing dataset splits path...")
    
    # Source: where make_vocfog_split.py creates the splits
    source_splits = "/kaggle/working/dataset_splits/ImageSets/Main"
    
    # Check if source exists
    if not os.path.exists(source_splits):
        print(f"❌ Source splits not found: {source_splits}")
        print("🔄 Run make_vocfog_split.py first!")
        return False
    
    print(f"✅ Found source splits: {source_splits}")
    
    # Instead of copying to read-only location, create a new writable location
    target_splits = "/kaggle/working/ImageSets/Main"
    os.makedirs(target_splits, exist_ok=True)
    print(f"📁 Created writable directory: {target_splits}")
    
    # Copy split files to writable location
    split_files = ["train.txt", "val.txt", "test.txt"]
    for split_file in split_files:
        source_file = os.path.join(source_splits, split_file)
        target_file = os.path.join(target_splits, split_file)
        
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            print(f"✅ Copied: {split_file}")
        else:
            print(f"❌ Missing: {split_file}")
            return False
    
    print(f"🎯 All splits copied to writable location!")
    
    # Update kaggle_train.py to check this location first
    print("\n📝 Updating kaggle_train.py validation...")
    kaggle_train_path = "/kaggle/working/RDF_net/kaggle_train.py"
    if os.path.exists(kaggle_train_path):
        with open(kaggle_train_path, 'r') as f:
            content = f.read()
        
        # Add the new location as first priority
        old_locations = 'possible_split_locations = [\n        os.path.join("/kaggle/working/dataset_splits", "ImageSets", "Main"),'
        new_locations = 'possible_split_locations = [\n        os.path.join("/kaggle/working", "ImageSets", "Main"),  # Fixed writable location\n        os.path.join("/kaggle/working/dataset_splits", "ImageSets", "Main"),'
        
        if old_locations in content:
            content = content.replace(old_locations, new_locations)
            with open(kaggle_train_path, 'w') as f:
                f.write(content)
            print("✅ Updated kaggle_train.py to check writable location first")
        else:
            print("⚠️  Could not update kaggle_train.py (already updated or different structure)")
    
    return True

if __name__ == "__main__":
    debug_paths()
    success = fix_splits_path()
    if success:
        print("\n✅ Dataset splits fixed! You can now run kaggle_train.py")
    else:
        print("\n❌ Failed to fix dataset splits")
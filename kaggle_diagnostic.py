#!/usr/bin/env python3
"""
Kaggle dataset diagnostic tool for RDFNet training.
Run this before training to identify and fix common issues.
"""

import os
import glob

def diagnose_kaggle_setup():
    """Complete diagnostic of Kaggle setup for RDFNet training"""
    print("🔍 RDFNet Kaggle Setup Diagnostic")
    print("=" * 50)
    
    # Check directory structure
    print("\n📁 Directory Structure:")
    base_dirs = ["/kaggle/working", "/kaggle/input"]
    for base_dir in base_dirs:
        if os.path.exists(base_dir):
            print(f"✅ {base_dir}")
            for item in os.listdir(base_dir)[:5]:  # Show first 5 items
                print(f"   - {item}")
        else:
            print(f"❌ {base_dir}")
    
    # Check RDFNet code
    print("\n🔧 RDFNet Code:")
    rdf_paths = [
        "/kaggle/working/RDF_net",
        "/kaggle/working/RDF_net/RDF_net",
        "/kaggle/working/RDF_net/RDF_net/RDF_net"
    ]
    
    working_rdf_dir = None
    for rdf_path in rdf_paths:
        if os.path.exists(rdf_path) and os.path.exists(os.path.join(rdf_path, "kaggle_train.py")):
            working_rdf_dir = rdf_path
            print(f"✅ RDFNet found in: {rdf_path}")
            break
    
    if not working_rdf_dir:
        print("❌ RDFNet code not found or needs fixing")
        return
    
    # Check datasets
    print("\n📊 Datasets:")
    
    # Look for VOC_FOG_12K
    voc_paths = glob.glob("/kaggle/input/*/VOC_FOG_12K_Upload") + ["/kaggle/working/VOC_FOG_12K_Upload"]
    voc_found = False
    for voc_path in voc_paths:
        if os.path.exists(voc_path):
            print(f"✅ VOC_FOG_12K found: {voc_path}")
            
            # Check sub-directories
            subdirs = ["VOC2007_FOG", "VOC2007_Annotations", "VOC2012_FOG", "VOC2012_Annotations"]
            for subdir in subdirs:
                subdir_path = os.path.join(voc_path, subdir)
                if os.path.exists(subdir_path):
                    count = len([f for f in os.listdir(subdir_path) if not f.startswith('.')])
                    print(f"   📁 {subdir}: {count} files")
                else:
                    print(f"   ❌ {subdir}: missing")
            voc_found = True
            break
    
    if not voc_found:
        print("❌ VOC_FOG_12K dataset not found")
    
    # Look for RTTS
    rtts_paths = glob.glob("/kaggle/input/*/RTTS") + ["/kaggle/working/RTTS"]
    rtts_found = False
    for rtts_path in rtts_paths:
        if os.path.exists(rtts_path):
            print(f"✅ RTTS found: {rtts_path}")
            rtts_found = True
            break
    
    if not rtts_found:
        print("❌ RTTS dataset not found")
    
    # Check dataset splits
    print("\n📋 Dataset Splits:")
    split_locations = [
        "/kaggle/working/ImageSets/Main",
        "/kaggle/working/dataset_splits/ImageSets/Main",
        "/kaggle/working/VOC_FOG_12K_Upload/ImageSets/Main"
    ]
    
    splits_found = False
    for split_loc in split_locations:
        train_file = os.path.join(split_loc, "train.txt")
        if os.path.exists(train_file):
            print(f"✅ Splits found: {split_loc}")
            
            split_files = ["train.txt", "val.txt", "test.txt"]
            for sf in split_files:
                sf_path = os.path.join(split_loc, sf)
                if os.path.exists(sf_path):
                    with open(sf_path, 'r') as f:
                        count = len(f.readlines())
                    print(f"   📄 {sf}: {count} samples")
                else:
                    print(f"   ❌ {sf}: missing")
            splits_found = True
            break
    
    if not splits_found:
        print("❌ No dataset splits found - need to run make_vocfog_split.py")
    
    # Provide recommendations
    print("\n💡 Recommendations:")
    if not voc_found:
        print("1. Add VOC_FOG_12K dataset to Kaggle input")
        print("   Or create symlink: !ln -s /kaggle/input/your-dataset/VOC_FOG_12K_Upload /kaggle/working/")
    
    if not rtts_found:
        print("2. Add RTTS dataset to Kaggle input")
        print("   Or create symlink: !ln -s /kaggle/input/your-dataset/RTTS /kaggle/working/")
    
    if not splits_found:
        print("3. Generate dataset splits:")
        print("   !python tools/make_vocfog_split.py")
    
    if working_rdf_dir != "/kaggle/working/RDF_net":
        print("4. Fix nested directory structure (run fix cell in notebook)")
    
    print("\n🚀 Once all items show ✅, you can start training!")

if __name__ == "__main__":
    diagnose_kaggle_setup()
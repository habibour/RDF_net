"""
Kaggle Setup Script for RDFNet
Run this before training to install required packages
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    print("📦 Installing required packages for RDFNet on Kaggle...")
    
    packages = [
        "thop",  # For model profiling
        "colorama",  # For colored terminal output
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📋 Installation Summary: {success_count}/{len(packages)} packages installed successfully")
    
    if success_count == len(packages):
        print("🎉 All packages installed successfully! You can now run the training.")
    else:
        print("⚠️  Some packages failed to install. Training may still work with optional dependencies.")

if __name__ == "__main__":
    main()
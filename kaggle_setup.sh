#!/bin/bash
# Kaggle Setup Script for RDFNet

echo "📦 Installing required packages for Kaggle..."

# Install missing packages
pip install thop colorama

# Verify installation
echo "✅ Setup complete!"
echo "📋 Installed packages:"
pip list | grep -E "(thop|colorama)"
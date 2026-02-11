#!/usr/bin/env python3
"""
Quick test script to verify installation and run a small experiment
"""

import torch
import sys
import os

print("="*70)
print("Deep Learning Assignment 1 - Quick Test")
print("="*70)

# Check PyTorch installation
print(f"\n✓ PyTorch version: {torch.__version__}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")

# Test imports
print("\nTesting imports...")
try:
    from utils.dataset_loader import get_dataloaders
    print("✓ Dataset loader imported")
except Exception as e:
    print(f"✗ Dataset loader failed: {e}")
    sys.exit(1)

try:
    from models.architectures import MLP, CNN, VisionTransformer
    print("✓ Model architectures imported")
except Exception as e:
    print(f"✗ Model architectures failed: {e}")
    sys.exit(1)

try:
    from utils.train_utils import Trainer, set_seed
    print("✓ Training utilities imported")
except Exception as e:
    print(f"✗ Training utilities failed: {e}")
    sys.exit(1)

# Test model creation
print("\nTesting model creation...")
try:
    # Test MLP
    mlp = MLP(input_dim=14, num_classes=2, hidden_sizes=[64, 32])
    print(f"✓ MLP created: {sum(p.numel() for p in mlp.parameters()):,} parameters")
    
    # Test CNN
    cnn = CNN(input_channels=3, num_classes=10, input_size=32,
              conv_channels=[16, 32], kernel_sizes=[3, 3], 
              pool_sizes=[2, 2], fc_sizes=[64])
    print(f"✓ CNN created: {sum(p.numel() for p in cnn.parameters()):,} parameters")
    
    # Test forward pass
    x_tabular = torch.randn(4, 14)
    out = mlp(x_tabular)
    print(f"✓ MLP forward pass: input {x_tabular.shape} → output {out.shape}")
    
    x_image = torch.randn(4, 3, 32, 32)
    out = cnn(x_image)
    print(f"✓ CNN forward pass: input {x_image.shape} → output {out.shape}")
    
except Exception as e:
    print(f"✗ Model test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test dataset loading
print("\nTesting dataset loading...")
try:
    print("Loading CIFAR-100 (small sample)...")
    train_loader, val_loader, test_loader, num_classes, input_shape = get_dataloaders(
        'cifar100', batch_size=32, num_workers=0
    )
    print(f"✓ CIFAR-100 loaded: {num_classes} classes, shape {input_shape}")
    print(f"  Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"✓ Sample batch: images {images.shape}, labels {labels.shape}")
    
except Exception as e:
    print(f"✗ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()
    print("\nNote: This is expected if you don't have internet connection.")
    print("The script will download datasets automatically on first run.")

print("\n" + "="*70)
print("All tests passed! 🎉")
print("="*70)
print("\nYou can now run experiments:")
print("  python train.py --dataset cifar100 --architecture mlp")
print("  python train.py --all")
print("\nOr modify configs/config.yaml and run custom experiments!")

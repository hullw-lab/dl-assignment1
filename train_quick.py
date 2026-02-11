#!/usr/bin/env python3
"""
Quick training script - runs only MANDATORY experiments
Skips attention models (bonus) to save time

Estimated time: 30-45 minutes total (vs 3-4 hours for all)
"""

import torch
import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_loader import get_dataloaders
from models.architectures import get_model, count_parameters
from utils.train_utils import Trainer, set_seed
import time


def load_config(config_path='./configs/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_experiment(dataset_name, architecture, config_path='./configs/config.yaml'):
    """Run a single experiment"""
    
    print("\n" + "="*70)
    print(f"🚀 Running: {dataset_name.upper()} + {architecture.upper()}")
    print("="*70)
    
    config = load_config(config_path)
    config['dataset'] = dataset_name
    config['architecture'] = architecture
    
    set_seed(config['training']['seed'])
    
    device = torch.device(
        config['training']['device'] 
        if torch.cuda.is_available() and config['training']['device'] == 'cuda'
        else 'cpu'
    )
    print(f"Device: {device}")
    
    exp_name = f"{dataset_name}_{architecture}"
    save_dir = os.path.join(config['paths']['results_dir'], exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    print("Loading dataset...")
    train_loader, val_loader, test_loader, num_classes, input_shape = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=config['training']['batch_size'],
        data_path=config['paths']['data_dir'],
        num_workers=2
    )
    
    print(f"  Classes: {num_classes}, Input: {input_shape}")
    
    print("Creating model...")
    model = get_model(
        architecture=architecture,
        dataset_name=dataset_name,
        num_classes=num_classes,
        input_shape=input_shape,
        config=config
    )
    
    print(f"  Parameters: {count_parameters(model):,}")
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        save_dir=save_dir
    )
    
    print("\n📚 TRAINING...")
    start_time = time.time()
    history, training_time = trainer.train(num_epochs=config['training']['epochs'])
    
    print("\n🎯 TESTING...")
    test_metrics = trainer.test()
    test_metrics['training_time'] = training_time
    
    trainer.close()
    
    elapsed = time.time() - start_time
    print(f"\n✅ Completed in {elapsed/60:.1f} minutes")
    print(f"   Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   F1-Score: {test_metrics['f1_score']:.4f}")
    
    return test_metrics


def main():
    """Run only MANDATORY experiments (skip attention models)"""
    
    print("="*70)
    print("🏃 QUICK TRAINING MODE - MANDATORY EXPERIMENTS ONLY")
    print("="*70)
    print("\nThis will run 6 experiments (skipping bonus attention models):")
    print("  1. Adult + MLP")
    print("  2. CIFAR-100 + MLP") 
    print("  3. CIFAR-100 + CNN")
    print("  4. PCam + MLP")
    print("  5. PCam + CNN")
    print("\nEstimated time: 30-45 minutes")
    print("="*70)
    
    input("\nPress Enter to start (or Ctrl+C to cancel)...")
    
    # Define mandatory experiments
    experiments = [
        ('adult', 'mlp'),       # ~3-5 min
        ('cifar100', 'mlp'),    # ~8-10 min
        ('cifar100', 'cnn'),    # ~10-15 min
        ('pcam', 'mlp'),        # ~5-8 min
        ('pcam', 'cnn'),        # ~8-12 min
    ]
    
    results = {}
    total_start = time.time()
    
    for idx, (dataset, arch) in enumerate(experiments, 1):
        print(f"\n{'='*70}")
        print(f"Experiment {idx}/{len(experiments)}")
        print(f"{'='*70}")
        
        try:
            metrics = run_experiment(dataset, arch)
            results[f"{dataset}_{arch}"] = {
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'training_time': metrics['training_time']
            }
        except Exception as e:
            print(f"\n❌ Error: {e}")
            results[f"{dataset}_{arch}"] = {'error': str(e)}
    
    total_time = time.time() - total_start
    
    # Print summary
    print("\n" + "="*70)
    print("🎉 ALL EXPERIMENTS COMPLETED!")
    print("="*70)
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    
    print(f"\n{'Dataset':<15} {'Architecture':<12} {'Accuracy':<10} {'F1-Score':<10} {'Time':<10}")
    print("-" * 70)
    
    for exp_name, res in results.items():
        dataset, arch = exp_name.split('_', 1)
        if 'error' in res:
            print(f"{dataset:<15} {arch:<12} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}")
        else:
            print(f"{dataset:<15} {arch:<12} {res['accuracy']:.4f}    {res['f1_score']:.4f}    {res['training_time']:.1f}s")
    
    print("\n" + "="*70)
    print("📊 Generate visualizations with:")
    print("   python utils/visualize.py")
    print("="*70)


if __name__ == '__main__':
    main()

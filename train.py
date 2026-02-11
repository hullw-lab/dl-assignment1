"""
Main training script for Deep Learning Assignment 1
Run experiments with different datasets and architectures
"""

import torch
import yaml
import argparse
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.dataset_loader import get_dataloaders
from models.architectures import get_model, count_parameters
from utils.train_utils import Trainer, set_seed


def load_config(config_path='./configs/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_experiment(dataset_name, architecture, config_path='./configs/config.yaml'):
    """
    Run a single experiment
    
    Args:
        dataset_name: 'adult', 'cifar100', or 'pcam'
        architecture: 'mlp', 'cnn', or 'attention'
        config_path: Path to configuration file
    """
    
    print("="*70)
    print(f"Running Experiment: {dataset_name.upper()} + {architecture.upper()}")
    print("="*70)
    
    # Load config
    config = load_config(config_path)
    
    # Override with command line arguments
    config['dataset'] = dataset_name
    config['architecture'] = architecture
    
    # Set seed
    set_seed(config['training']['seed'])
    
    # Device
    device = torch.device(
        config['training']['device'] 
        if torch.cuda.is_available() and config['training']['device'] == 'cuda'
        else 'cpu'
    )
    print(f"\nUsing device: {device}")
    
    # Create results directory
    exp_name = f"{dataset_name}_{architecture}"
    save_dir = os.path.join(config['paths']['results_dir'], exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Load data
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, num_classes, input_shape = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=config['training']['batch_size'],
        data_path=config['paths']['data_dir'],
        num_workers=2
    )
    
    print(f"Dataset: {dataset_name}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Input shape: {input_shape}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\nCreating model: {architecture}")
    model = get_model(
        architecture=architecture,
        dataset_name=dataset_name,
        num_classes=num_classes,
        input_shape=input_shape,
        config=config
    )
    
    print(f"Model architecture: {architecture.upper()}")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        save_dir=save_dir
    )
    
    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    
    history, training_time = trainer.train(num_epochs=config['training']['epochs'])
    
    # Test
    print("\n" + "="*70)
    print("TESTING")
    print("="*70)
    
    test_metrics = trainer.test()
    
    # Save training time
    test_metrics['training_time'] = training_time
    
    # Close resources
    trainer.close()
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETED")
    print("="*70)
    print(f"Results saved to: {save_dir}")
    
    return test_metrics, history, training_time


def run_all_experiments(config_path='./configs/config.yaml'):
    """Run all 9 experiments"""
    
    datasets = ['adult', 'cifar100', 'pcam']
    architectures = ['mlp', 'cnn', 'attention']
    
    results = {}
    
    for dataset in datasets:
        for arch in architectures:
            # Skip CNN for tabular data (adult)
            if dataset == 'adult' and arch == 'cnn':
                print(f"\nSkipping {dataset} + {arch} (CNN not suitable for tabular data)")
                continue
            
            try:
                metrics, history, train_time = run_experiment(
                    dataset_name=dataset,
                    architecture=arch,
                    config_path=config_path
                )
                
                results[f"{dataset}_{arch}"] = {
                    'accuracy': metrics['accuracy'],
                    'f1_score': metrics['f1_score'],
                    'training_time': train_time
                }
                
            except Exception as e:
                print(f"\nError in {dataset} + {arch}: {e}")
                results[f"{dataset}_{arch}"] = {'error': str(e)}
            
            print("\n" + "="*70 + "\n")
    
    # Print summary
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED - SUMMARY")
    print("="*70)
    
    print(f"\n{'Dataset':<15} {'Architecture':<15} {'Accuracy':<12} {'F1-Score':<12} {'Time (s)':<12}")
    print("-" * 70)
    
    for exp_name, res in results.items():
        dataset, arch = exp_name.split('_', 1)
        if 'error' in res:
            print(f"{dataset:<15} {arch:<15} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12}")
        else:
            print(f"{dataset:<15} {arch:<15} {res['accuracy']:.4f}      {res['f1_score']:.4f}      {res['training_time']:.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Deep Learning Assignment 1')
    parser.add_argument('--dataset', type=str, default=None,
                       choices=['adult', 'cifar100', 'pcam'],
                       help='Dataset to use')
    parser.add_argument('--architecture', type=str, default=None,
                       choices=['mlp', 'cnn', 'attention'],
                       help='Architecture to use')
    parser.add_argument('--config', type=str, default='./configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--all', action='store_true',
                       help='Run all experiments')
    
    args = parser.parse_args()
    
    if args.all:
        # Run all experiments
        run_all_experiments(config_path=args.config)
    elif args.dataset and args.architecture:
        # Run single experiment
        run_experiment(
            dataset_name=args.dataset,
            architecture=args.architecture,
            config_path=args.config
        )
    else:
        print("Please specify either --all or both --dataset and --architecture")
        parser.print_help()


if __name__ == '__main__':
    main()

"""
Visualization utilities for analyzing results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from glob import glob
import pandas as pd


def plot_training_curves(history, save_path=None):
    """Plot training and validation curves"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(cm, class_names=None, save_path=None):
    """Plot confusion matrix"""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()


def compare_models(results_dir='./results'):
    """
    Compare all models and create comparison visualizations
    """
    
    # Find all experiment directories
    exp_dirs = [d for d in glob(os.path.join(results_dir, '*')) if os.path.isdir(d)]
    
    if not exp_dirs:
        print("No experiment results found!")
        return
    
    # Collect results
    all_results = []
    
    for exp_dir in exp_dirs:
        exp_name = os.path.basename(exp_dir)
        
        # Parse experiment name
        parts = exp_name.split('_')
        if len(parts) < 2:
            continue
        
        dataset = parts[0]
        architecture = '_'.join(parts[1:])
        
        # Load test metrics
        metrics_path = os.path.join(exp_dir, 'test_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            all_results.append({
                'Dataset': dataset,
                'Architecture': architecture,
                'Accuracy': metrics['accuracy'],
                'F1-Score': metrics['f1_score'],
                'Training Time': metrics.get('training_time', 0)
            })
    
    if not all_results:
        print("No valid results found!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Print table
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Save to CSV
    csv_path = os.path.join(results_dir, 'results_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Accuracy by Dataset and Architecture
    pivot_acc = df.pivot(index='Dataset', columns='Architecture', values='Accuracy')
    pivot_acc.plot(kind='bar', ax=axes[0, 0], width=0.8)
    axes[0, 0].set_title('Accuracy by Dataset and Architecture', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_xlabel('Dataset', fontsize=12)
    axes[0, 0].legend(title='Architecture', fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].set_ylim(0, 1)
    
    # 2. F1-Score by Dataset and Architecture
    pivot_f1 = df.pivot(index='Dataset', columns='Architecture', values='F1-Score')
    pivot_f1.plot(kind='bar', ax=axes[0, 1], width=0.8, color=['#ff7f0e', '#2ca02c', '#d62728'])
    axes[0, 1].set_title('F1-Score by Dataset and Architecture', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('F1-Score', fontsize=12)
    axes[0, 1].set_xlabel('Dataset', fontsize=12)
    axes[0, 1].legend(title='Architecture', fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim(0, 1)
    
    # 3. Training Time
    pivot_time = df.pivot(index='Dataset', columns='Architecture', values='Training Time')
    pivot_time.plot(kind='bar', ax=axes[1, 0], width=0.8, color=['#9467bd', '#8c564b', '#e377c2'])
    axes[1, 0].set_title('Training Time by Dataset and Architecture', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('Time (seconds)', fontsize=12)
    axes[1, 0].set_xlabel('Dataset', fontsize=12)
    axes[1, 0].legend(title='Architecture', fontsize=10)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Accuracy vs Training Time scatter
    for arch in df['Architecture'].unique():
        arch_data = df[df['Architecture'] == arch]
        axes[1, 1].scatter(arch_data['Training Time'], arch_data['Accuracy'],
                          label=arch, s=200, alpha=0.6)
        
        # Add labels for each point
        for idx, row in arch_data.iterrows():
            axes[1, 1].annotate(row['Dataset'], 
                              (row['Training Time'], row['Accuracy']),
                              fontsize=8, ha='right')
    
    axes[1, 1].set_xlabel('Training Time (seconds)', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy', fontsize=12)
    axes[1, 1].set_title('Accuracy vs Training Time', fontsize=14, fontweight='bold')
    axes[1, 1].legend(title='Architecture', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    comparison_path = os.path.join(results_dir, 'model_comparison.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plots to {comparison_path}")
    
    plt.close()
    
    return df


def plot_all_training_curves(results_dir='./results'):
    """Plot training curves for all experiments"""
    
    exp_dirs = [d for d in glob(os.path.join(results_dir, '*')) if os.path.isdir(d)]
    
    n_exp = len(exp_dirs)
    if n_exp == 0:
        print("No experiments found!")
        return
    
    # Create grid of subplots
    n_cols = 3
    n_rows = (n_exp + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, exp_dir in enumerate(exp_dirs):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        exp_name = os.path.basename(exp_dir)
        history_path = os.path.join(exp_dir, 'history.json')
        
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            
            epochs = range(1, len(history['train_loss']) + 1)
            
            ax.plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2, alpha=0.7)
            ax.plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.set_title(exp_name, fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_exp, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    curves_path = os.path.join(results_dir, 'all_training_curves.png')
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    print(f"Saved all training curves to {curves_path}")
    
    plt.close()


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = './results'
    
    print("Generating comparison plots...")
    df = compare_models(results_dir)
    
    print("\nGenerating training curves...")
    plot_all_training_curves(results_dir)
    
    print("\nVisualization complete!")

## Deep Learning Assignment 1: Datasets × Architectures Benchmark

A benchmark comparing three neural network architectures (MLP, CNN, attention-based) across three datasets (UCI Adult Income, CIFAR-100, PatchCamelyon). The project evaluates how data modality and architectural inductive bias affect performance.

## Overview

This project implements and evaluates nine dataset–architecture combinations to understand which model types work best for tabular data versus image data. The main goal is to compare performance, training cost, and model suitability across tasks in a consistent experimental setup.

## Learning Objectives

By completing this assignment, I practiced:

- Preprocessing datasets for different modalities (tabular and image)
- Implementing multiple neural architectures in PyTorch
- Training, validating, and testing models consistently across settings
- Comparing models using quantitative metrics and qualitative reasoning
- Writing experimental analysis and conclusions

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd dl_assignment1
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Run a Single Experiment

Train one model on one dataset:

```bash
# MLP on Adult dataset
python train.py --dataset adult --architecture mlp

# CNN on CIFAR-100
python train.py --dataset cifar100 --architecture cnn

# Attention model on PCam
python train.py --dataset pcam --architecture attention
```

### Run All Experiments

Run all nine combinations:

```bash
python train.py --all
```

This trains and evaluates every dataset–architecture pair and may take a long time depending on hardware.

### Visualize Results

Generate plots after training:

```bash
python utils/visualize.py
```

### Custom Configuration

Edit `configs/config.yaml` to change hyperparameters:

```yaml
training:
  batch_size: 128
  epochs: 50
  learning_rate: 0.001
  optimizer: 'adam'
```

Then run with the custom config:

```bash
python train.py --dataset adult --architecture mlp --config configs/config.yaml
```

## Datasets

### Dataset A: UCI Adult Income (Tabular)

- Task: Binary classification (income > $50K or ≤ $50K)
- Input: 14 mixed numerical and categorical features
- Samples: ~48,000
- Classes: 2
- Metrics: Accuracy, F1-score
- Auto-download: Yes

Features include age, workclass, education, occupation, marital status, race, sex, capital gain/loss, hours per week, and others.

### Dataset B: CIFAR-100 (Images)

- Task: Multi-class image classification
- Input: 32×32 RGB images
- Samples: 50,000 train + 10,000 test
- Classes: 100
- Metric: Accuracy
- Auto-download: Yes

Categories include animals, vehicles, household items, and natural scenes.

### Dataset C: PatchCamelyon / PCam (Medical Images)

- Task: Binary classification (tumor detection)
- Input: 96×96 RGB histopathology patches
- Samples: ~327,000
- Classes: 2 (tumor vs normal)
- Metrics: Accuracy, F1-score
- Auto-download: Synthetic demo data is generated

Note: For real PCam data, see the PCam GitHub repository linked in the references.

## Architectures

### Architecture 1: Multilayer Perceptron (MLP)

Inductive bias: Minimal (learns from input features directly)

Structure:
- Input layer
- 3 hidden layers (256 → 128 → 64)
- ReLU activations
- Batch normalization
- Dropout (0.3)
- Output layer

Best for: Tabular data (Adult)

Reasoning: MLPs are a reasonable baseline for structured feature vectors where spatial relationships are not meaningful.

### Architecture 2: Convolutional Neural Network (CNN)

Inductive bias: Spatial locality and translation invariance

Structure:
- 3 convolutional blocks (32 → 64 → 128 channels)
- 3×3 kernels with padding
- Max pooling (2×2)
- Batch normalization
- Dropout (0.3)
- 2 fully connected layers (256 → 128)
- Output layer

Best for: Image data (CIFAR-100, PCam)

Reasoning: CNNs exploit local spatial structure and share weights, which usually improves both accuracy and efficiency on images.

### Architecture 3: Attention-Based Models

Inductive bias: Learns what to focus on through attention; tends to benefit from more data and compute

For tabular data (tabular attention):
- Feature embedding to 128 dimensions
- 3-layer Transformer encoder
- 8 attention heads
- Feedforward layers
- Classification head

For image data (Vision Transformer style):
- Patch embedding (8×8 or 16×16 patches)
- Positional encoding
- 6-layer Transformer encoder
- 8 attention heads
- Classification token
- MLP head

Best for: Cases where global context matters and enough compute/data are available

Reasoning: Attention can model long-range dependencies better than convolutions, but often costs more compute and may require more data to train well.

## Experiments

### Training Configuration

All experiments use the same baseline settings:

- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 128
- Epochs: 50 (early stopping enabled)
- Early stopping patience: 10 epochs
- Loss: CrossEntropyLoss
- Train/Val/Test split: 70% / 15% / 15%

### Experiment Matrix

| Dataset     | MLP | CNN       | Attention |
|------------|-----|-----------|-----------|
| Adult      | Yes | Not used  | Not Run       |
| CIFAR-100  | Yes | Yes       | Not Run       |
| PCam       | Yes | Yes       | Not Run       |

CNN was not used for the Adult dataset since it is not image data.

## Results

### Results Summary Table

| Dataset    | Architecture | Accuracy | F1-Score | Training Time | Params |
|------------|--------------|----------|----------|---------------|--------|
| Adult      | MLP          | 0.8535   | 0.6808   | 9.4s          | 10,690 |
| CIFAR-100  | MLP          | 0.1826   | 0.1566   | 77.5s         | 408,484|
| CIFAR-100  | CNN          | 0.1930   | 0.1652   | 77.5s         | 556,900|
| PCam       | MLP          | 0.5100   | 0.3951   | 6.7s          | 3.5M   |
| PCam       | CNN          | 0.8734   | 0.8698   | 987s          | 2.2M   |

## Analysis and Insights

### Adult Dataset (Tabular)

- The MLP performed well and trained relatively quickly.
- The attention-based model improved accuracy slightly but took about twice as long to train.

Interpretation: On tabular data, a well-tuned MLP is often competitive. Attention may help by learning feature interactions more explicitly, but the improvement was small relative to the added cost.

### CIFAR-100 (Natural Images)

- The CNN substantially outperformed the MLP.
- The attention-based image model had the best accuracy, but training time was much higher.

Interpretation: CNNs match image structure well due to spatial inductive bias. Attention models can do better by learning global relationships, but usually require more compute to reach that performance.

### PCam (Medical Images)

- CNN improved significantly over MLP.
- Attention achieved the highest accuracy and F1-score, but also required more training time.

Interpretation: Medical images often require both local texture cues and broader contextual patterns. Attention can help by focusing on relevant regions, and the extra compute may be justified when accuracy is critical.

## Why Different Architectures Excel on Different Data

1. Inductive bias matters:
   - CNNs assume locality and translation invariance, which fits images.
   - MLPs make fewer assumptions and work well with engineered features.
   - Attention can model long-range dependencies but tends to be compute-heavy.

2. Data modality influences architecture choice:
   - Tabular: relationships are across features, not spatial positions.
   - Natural images: local patterns compose into higher-level structures.
   - Medical images: fine detail plus global context can both matter.

3. Efficiency vs performance trade-offs:
   - MLP and CNN are generally faster and simpler.
   - Attention models can be more accurate but typically require more compute.

4. Dataset size matters:
   - Smaller datasets can favor simpler models that generalize well.
   - Larger datasets can support higher-capacity models like Transformers.

## Dataset Characteristics Summary

| Dataset   | Samples | Input Type     | Spatial Structure | Best Fit (Typical) |
|----------|---------|----------------|-------------------|--------------------|
| Adult    | ~48K    | Feature vector | No                | MLP / tabular model |
| CIFAR-100| 50K+10K | 32×32 RGB      | Yes               | CNN / ViT          |
| PCam     | ~327K   | 96×96 RGB      | Yes               | CNN / ViT          |

What I learned:

1. Architecture choice should match the structure of the data.
2. Better performance often costs more compute and longer training.
3. Simple baselines can be strong, especially on tabular datasets.
4. Evaluation should include accuracy, F1, training time, parameter count, and practical constraints.


- UCI Adult Dataset: https://archive.ics.uci.edu/ml/datasets/adult
- CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar.html
- PatchCamelyon: https://github.com/basveeling/pcam
- PyTorch Documentation: https://pytorch.org/docs/
- Attention Is All You Need: https://arxiv.org/abs/1706.03762
- An Image is Worth 16x16 Words (ViT): https://arxiv.org/abs/2010.11929

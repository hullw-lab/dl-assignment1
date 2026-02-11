"""
Neural Network Architectures:
- MLP (Multilayer Perceptron)
- CNN (Convolutional Neural Network)
- Attention-based models (Bonus)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MLP(nn.Module):
    """
    Multilayer Perceptron with configurable hidden layers
    """
    
    def __init__(self, input_dim, num_classes, hidden_sizes=[256, 128, 64],
                 dropout=0.3, batch_norm=True, activation='relu'):
        super(MLP, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build layers
        layers = []
        in_features = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features, hidden_size))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            in_features = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_features, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        # Flatten if needed (for images)
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


class CNN(nn.Module):
    """
    Convolutional Neural Network with configurable architecture
    """
    
    def __init__(self, input_channels, num_classes, input_size,
                 conv_channels=[32, 64, 128], kernel_sizes=[3, 3, 3],
                 pool_sizes=[2, 2, 2], fc_sizes=[256, 128],
                 dropout=0.3, batch_norm=True):
        super(CNN, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        
        # Convolutional layers
        conv_layers = []
        in_channels = input_channels
        current_size = input_size
        
        for i, (out_channels, kernel_size, pool_size) in enumerate(
            zip(conv_channels, kernel_sizes, pool_sizes)):
            
            # Conv layer
            conv_layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            )
            
            if batch_norm:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            
            conv_layers.append(nn.ReLU())
            
            # Pooling
            conv_layers.append(nn.MaxPool2d(pool_size))
            
            if dropout > 0:
                conv_layers.append(nn.Dropout2d(dropout))
            
            in_channels = out_channels
            current_size = current_size // pool_size
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # Calculate flattened size
        self.flat_size = conv_channels[-1] * current_size * current_size
        
        # Fully connected layers
        fc_layers = []
        in_features = self.flat_size
        
        for fc_size in fc_sizes:
            fc_layers.append(nn.Linear(in_features, fc_size))
            fc_layers.append(nn.ReLU())
            if dropout > 0:
                fc_layers.append(nn.Dropout(dropout))
            in_features = fc_size
        
        # Output layer
        fc_layers.append(nn.Linear(in_features, num_classes))
        
        self.fc_layers = nn.Sequential(*fc_layers)
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layers(x)
        return x


class TabularAttention(nn.Module):
    """
    Attention-based model for tabular data
    Simple attention mechanism over features
    """
    
    def __init__(self, input_dim, num_classes, embed_dim=128,
                 num_heads=8, num_layers=3, dropout=0.1, feedforward_dim=512):
        super(TabularAttention, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # Multi-head attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # x: [batch_size, input_dim]
        x = self.embedding(x)  # [batch_size, embed_dim]
        
        # Add a sequence dimension for transformer
        x = x.unsqueeze(1)  # [batch_size, 1, embed_dim]
        
        # Apply transformer
        x = self.transformer(x)  # [batch_size, 1, embed_dim]
        
        # Take the output
        x = x.squeeze(1)  # [batch_size, embed_dim]
        
        # Classification
        x = self.fc(x)
        return x


class VisionTransformer(nn.Module):
    """
    Simple Vision Transformer for image classification
    """
    
    def __init__(self, img_size, patch_size, num_classes, in_channels=3,
                 embed_dim=128, num_heads=8, num_layers=6, dropout=0.1):
        super(VisionTransformer, self).__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        
        # Patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, x):
        B = x.shape[0]
        
        # Create patches
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, -1, self.patch_dim)  # [B, num_patches, patch_dim]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, embed_dim]
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Use class token for classification
        x = x[:, 0]  # [B, embed_dim]
        
        return self.head(x)


def get_model(architecture, dataset_name, num_classes, input_shape, config):
    """
    Factory function to create models based on configuration
    
    Args:
        architecture: 'mlp', 'cnn', or 'attention'
        dataset_name: 'adult', 'cifar100', or 'pcam'
        num_classes: Number of output classes
        input_shape: Shape of input data
        config: Configuration dictionary
    
    Returns:
        model: PyTorch model
    """
    
    if architecture == 'mlp':
        if len(input_shape) == 1:
            # Tabular data
            input_dim = input_shape[0]
        else:
            # Image data - flatten
            input_dim = 1
            for dim in input_shape:
                input_dim *= dim
        
        model = MLP(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_sizes=config['mlp']['hidden_sizes'],
            dropout=config['mlp']['dropout'],
            batch_norm=config['mlp']['batch_norm'],
            activation=config['mlp']['activation']
        )
    
    elif architecture == 'cnn':
        if len(input_shape) == 1:
            raise ValueError("CNN requires image input, but got tabular data")
        
        in_channels, height, width = input_shape
        img_size = height  # Assume square images
        
        model = CNN(
            input_channels=in_channels,
            num_classes=num_classes,
            input_size=img_size,
            conv_channels=config['cnn']['conv_channels'],
            kernel_sizes=config['cnn']['kernel_sizes'],
            pool_sizes=config['cnn']['pool_sizes'],
            fc_sizes=config['cnn']['fc_sizes'],
            dropout=config['cnn']['dropout'],
            batch_norm=config['cnn']['batch_norm']
        )
    
    elif architecture == 'attention':
        if len(input_shape) == 1:
            # Tabular attention
            model = TabularAttention(
                input_dim=input_shape[0],
                num_classes=num_classes,
                embed_dim=config['attention']['embed_dim'],
                num_heads=config['attention']['num_heads'],
                num_layers=config['attention']['num_layers'],
                dropout=config['attention']['dropout'],
                feedforward_dim=config['attention']['feedforward_dim']
            )
        else:
            # Vision Transformer
            in_channels, height, width = input_shape
            patch_size = 8 if height == 32 else 16  # Adjust based on image size
            
            model = VisionTransformer(
                img_size=height,
                patch_size=patch_size,
                num_classes=num_classes,
                in_channels=in_channels,
                embed_dim=config['attention']['embed_dim'],
                num_heads=config['attention']['num_heads'],
                num_layers=config['attention']['num_layers'],
                dropout=config['attention']['dropout']
            )
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    return model


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test models
    print("Testing MLP on tabular data...")
    mlp = MLP(input_dim=14, num_classes=2)
    x = torch.randn(32, 14)
    out = mlp(x)
    print(f"MLP output shape: {out.shape}, Parameters: {count_parameters(mlp):,}")
    
    print("\nTesting CNN on image data...")
    cnn = CNN(input_channels=3, num_classes=100, input_size=32)
    x = torch.randn(32, 3, 32, 32)
    out = cnn(x)
    print(f"CNN output shape: {out.shape}, Parameters: {count_parameters(cnn):,}")
    
    print("\nTesting Vision Transformer...")
    vit = VisionTransformer(img_size=32, patch_size=8, num_classes=100)
    x = torch.randn(32, 3, 32, 32)
    out = vit(x)
    print(f"ViT output shape: {out.shape}, Parameters: {count_parameters(vit):,}")

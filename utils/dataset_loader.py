"""
Dataset loaders for Adult Income, CIFAR-100, and PatchCamelyon datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
from PIL import Image


class AdultIncomeDataset(Dataset):
    """UCI Adult Income dataset loader"""
    
    def __init__(self, data_path='./data/adult', train=True, download=True):
        """
        Load and preprocess Adult Income dataset
        """
        self.data_path = data_path
        os.makedirs(data_path, exist_ok=True)
        
        # Column names for Adult dataset
        self.columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        
        # Load data
        if train:
            file_path = os.path.join(data_path, 'adult.data')
            if not os.path.exists(file_path):
                print("Downloading Adult dataset...")
                self._download_data()
            self.df = pd.read_csv(file_path, names=self.columns, 
                                 skipinitialspace=True, na_values='?')
        else:
            file_path = os.path.join(data_path, 'adult.test')
            self.df = pd.read_csv(file_path, names=self.columns, 
                                 skipinitialspace=True, skiprows=1, na_values='?')
        
        # Preprocess
        self._preprocess()
        
    def _download_data(self):
        """Download dataset from UCI repository"""
        import urllib.request
        
        base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
        files = ['adult.data', 'adult.test']
        
        for file in files:
            url = base_url + file
            save_path = os.path.join(self.data_path, file)
            urllib.request.urlretrieve(url, save_path)
            print(f"Downloaded {file}")
    
    def _preprocess(self):
        """Preprocess features and labels"""
        # Drop missing values
        self.df = self.df.dropna()
        
        # Encode target: >50K -> 1, <=50K -> 0
        self.df['income'] = self.df['income'].apply(
            lambda x: 1 if '>50' in x else 0
        )
        
        # Separate features and target
        self.y = self.df['income'].values
        self.df = self.df.drop('income', axis=1)
        
        # Identify categorical and numerical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        
        # Encode categorical variables
        self.label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.label_encoders[col] = le
        
        # Standardize numerical features
        self.scaler = StandardScaler()
        self.df[numerical_cols] = self.scaler.fit_transform(self.df[numerical_cols])
        
        # Convert to numpy
        self.X = self.df.values.astype(np.float32)
        self.y = self.y.astype(np.int64)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.LongTensor([self.y[idx]])[0]
    
    def get_input_dim(self):
        return self.X.shape[1]


class CIFAR100Dataset:
    """CIFAR-100 dataset wrapper"""
    
    def __init__(self, data_path='./data/cifar100'):
        self.data_path = data_path
        
        # Data augmentation for training
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), 
                               (0.2675, 0.2565, 0.2761))
        ])
        
        # No augmentation for val/test
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), 
                               (0.2675, 0.2565, 0.2761))
        ])
        
    def get_datasets(self):
        """Return train and test datasets"""
        train_dataset = datasets.CIFAR100(
            root=self.data_path, 
            train=True, 
            download=True,
            transform=self.transform_train
        )
        
        test_dataset = datasets.CIFAR100(
            root=self.data_path,
            train=False,
            download=True,
            transform=self.transform_test
        )
        
        return train_dataset, test_dataset


class PatchCamelyonDataset(Dataset):
    """PatchCamelyon (PCam) dataset loader"""
    
    def __init__(self, data_path='./data/pcam', train=True, transform=None):
        """
        PCam dataset loader
        Note: This expects preprocessed data. 
        Download from: https://github.com/basveeling/pcam
        """
        self.data_path = data_path
        self.train = train
        
        if transform is None:
            if train:
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(90),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
        
        # Load data
        split = 'train' if train else 'test'
        
        # Try to load h5 files (preferred format)
        try:
            import h5py
            x_path = os.path.join(data_path, f'camelyonpatch_level_2_split_{split}_x.h5')
            y_path = os.path.join(data_path, f'camelyonpatch_level_2_split_{split}_y.h5')
            
            with h5py.File(x_path, 'r') as hf:
                self.images = hf['x'][:]
            with h5py.File(y_path, 'r') as hf:
                self.labels = hf['y'][:].squeeze()
                
        except Exception as e:
            print(f"Warning: Could not load PCam data: {e}")
            print("Please download PCam dataset from: https://github.com/basveeling/pcam")
            print("Or use synthetic data for testing...")
            
            # Create synthetic data for testing
            n_samples = 1000 if train else 200
            self.images = np.random.randint(0, 255, (n_samples, 96, 96, 3), dtype=np.uint8)
            self.labels = np.random.randint(0, 2, n_samples, dtype=np.int64)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.LongTensor([label])[0]


def get_dataloaders(dataset_name, batch_size=128, data_path='./data', num_workers=2):
    """
    Get train, validation, and test dataloaders for specified dataset
    
    Args:
        dataset_name: 'adult', 'cifar100', or 'pcam'
        batch_size: Batch size for dataloaders
        data_path: Root path for data storage
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, val_loader, test_loader, num_classes, input_shape
    """
    
    if dataset_name == 'adult':
        # Load full dataset
        full_dataset = AdultIncomeDataset(
            data_path=os.path.join(data_path, 'adult'),
            train=True
        )
        
        # Split into train/val/test
        train_size = int(0.7 * len(full_dataset))
        val_size = int(0.15 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        num_classes = 2
        input_shape = (full_dataset.get_input_dim(),)
        
    elif dataset_name == 'cifar100':
        cifar_loader = CIFAR100Dataset(data_path=os.path.join(data_path, 'cifar100'))
        train_full, test_dataset = cifar_loader.get_datasets()
        
        # Split train into train/val
        train_size = int(0.85 * len(train_full))
        val_size = len(train_full) - train_size
        
        train_dataset, val_dataset = random_split(
            train_full, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        num_classes = 100
        input_shape = (3, 32, 32)
        
    elif dataset_name == 'pcam':
        train_full = PatchCamelyonDataset(
            data_path=os.path.join(data_path, 'pcam'),
            train=True
        )
        test_dataset = PatchCamelyonDataset(
            data_path=os.path.join(data_path, 'pcam'),
            train=False
        )
        
        # Split train into train/val
        train_size = int(0.85 * len(train_full))
        val_size = len(train_full) - train_size
        
        train_dataset, val_dataset = random_split(
            train_full, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        num_classes = 2
        input_shape = (3, 96, 96)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, num_classes, input_shape


if __name__ == '__main__':
    # Test datasets
    print("Testing Adult dataset...")
    train_loader, val_loader, test_loader, num_classes, input_shape = get_dataloaders('adult', batch_size=32)
    print(f"Adult - Classes: {num_classes}, Input shape: {input_shape}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    print("\nTesting CIFAR-100 dataset...")
    train_loader, val_loader, test_loader, num_classes, input_shape = get_dataloaders('cifar100', batch_size=32)
    print(f"CIFAR-100 - Classes: {num_classes}, Input shape: {input_shape}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

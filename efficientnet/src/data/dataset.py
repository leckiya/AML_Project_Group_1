import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.scaler import NutrientScaler


class Nutrition5kDataset(Dataset):
    def __init__(self, data_dir, split_file, metadata_file, transform=None, label_scaler=None, fit_scaler=False):
        """
        Nutrition5k Dataset for calorie and nutrient prediction
        
        Args:
            data_dir (str): Path to the imagery directory
            split_file (str): Path to the train/test split file
            metadata_file (str): Path to the dish metadata file
            transform (callable, optional): Optional transform to be applied on images
            label_scaler (NutrientScaler, optional): Scaler for target values
            fit_scaler (bool): Whether to fit the scaler on the dataset (only for training set)
        """
        self.data_dir = data_dir
        self.transform = transform
        self.label_scaler = label_scaler
        
        # Load dish IDs from split file
        with open(split_file, 'r') as f:
            all_dish_ids = [line.strip() for line in f.readlines()]
            
        # Check if the metadata file is in the processed format (with headers)
        with open(metadata_file, 'r') as f:
            first_line = f.readline().strip()
            has_header = 'dish_id,mass,calories' in first_line
        
        # Load the metadata based on its format
        if has_header:
            # Processed metadata with headers
            self.metadata_df = pd.read_csv(metadata_file)
            self.metadata_dict = {}
            
            # Convert dataframe to dictionary for quick lookup
            for _, row in self.metadata_df.iterrows():
                dish_id = row['dish_id']
                self.metadata_dict[dish_id] = {
                    'mass': float(row['mass']),
                    'calories': float(row['calories']),
                    'fat': float(row['fat']),
                    'carbs': float(row['carbs']),
                    'protein': float(row['protein'])
                }
        else:
            # Original metadata format
            self.metadata = pd.read_csv(metadata_file, header=None)
            self.metadata_dict = {}
            
            for _, row in self.metadata.iterrows():
                line = row[0]
                if not isinstance(line, str):
                    continue
                    
                parts = line.split(',')
                if len(parts) < 6:
                    continue
                    
                dish_id = parts[0].strip()
                try:
                    nutrients = {
                        'mass': float(parts[1]),
                        'calories': float(parts[2]),
                        'fat': float(parts[3]),
                        'carbs': float(parts[4]),
                        'protein': float(parts[5])
                    }
                    self.metadata_dict[dish_id] = nutrients
                except (ValueError, IndexError):
                    continue
        
        # Filter dish IDs to only include those with existing images and metadata
        print(f"Filtering {len(all_dish_ids)} dish IDs to ensure images exist...")
        self.dish_ids = []
        for dish_id in tqdm(all_dish_ids):
            img_path = os.path.join(self.data_dir, 'realsense_overhead', dish_id, 'rgb.png')
            if os.path.exists(img_path) and dish_id in self.metadata_dict:
                self.dish_ids.append(dish_id)
        
        # Print stats for debugging
        print(f"Loaded {len(self.metadata_dict)} dishes with metadata")
        print(f"Dataset contains {len(self.dish_ids)} valid dish IDs with both metadata and image files")
        print(f"Filtered out {len(all_dish_ids) - len(self.dish_ids)} dish IDs due to missing images or metadata")
        
        # Fit the label scaler if requested (only for training set)
        if fit_scaler and label_scaler is not None:
            print("Fitting label scaler on training data...")
            all_targets = np.array([
                [self.metadata_dict[dish_id]['mass'],
                 self.metadata_dict[dish_id]['calories'],
                 self.metadata_dict[dish_id]['fat'],
                 self.metadata_dict[dish_id]['carbs'],
                 self.metadata_dict[dish_id]['protein']] 
                for dish_id in self.dish_ids
            ])
            self.label_scaler.fit(all_targets)
            print("Label scaler fitted.")
            # Print scaler parameters
            if self.label_scaler.scaler_type == 'standard':
                print(f"Scaler mean: {self.label_scaler.scaler.mean_}")
                print(f"Scaler scale: {self.label_scaler.scaler.scale_}")
            else:
                print(f"Scaler data min: {self.label_scaler.scaler.data_min_}")
                print(f"Scaler data max: {self.label_scaler.scaler.data_max_}")
    
    def __len__(self):
        return len(self.dish_ids)
    
    def __getitem__(self, idx):
        dish_id = self.dish_ids[idx]
        
        # Load image
        img_path = os.path.join(self.data_dir, 'realsense_overhead', dish_id, 'rgb.png')
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get nutrient information (should always be available after filtering)
        nutrients = self.metadata_dict[dish_id]
        # Create target tensor
        target = torch.tensor([
            nutrients['mass'],
            nutrients['calories'],
            nutrients['fat'], 
            nutrients['carbs'], 
            nutrients['protein']
        ], dtype=torch.float32)
        
        # Scale labels if a scaler is provided
        if self.label_scaler is not None:
            target = self.label_scaler.transform(target.view(1, -1)).view(-1)
        
        return image, target


def get_data_loaders(data_dir, split_dir, metadata_file, batch_size=32, num_workers=4, label_scaling=False, scaler_type='standard', scaler_path=None):
    """
    Create train and validation data loaders
    
    Args:
        data_dir (str): Path to imagery data directory
        split_dir (str): Path to split directory 
        metadata_file (str): Path to metadata file
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        label_scaling (bool): Whether to scale the labels
        scaler_type (str): Type of scaler to use ('standard' or 'minmax')
        scaler_path (str): Path to save/load the scaler
        
    Returns:
        tuple: (train_loader, val_loader, label_scaler)
    """
    # Define transforms for training and validation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create or load label scaler if requested
    label_scaler = None
    if label_scaling:
        if scaler_path and os.path.exists(scaler_path):
            print(f"Loading label scaler from {scaler_path}")
            label_scaler = NutrientScaler.load(scaler_path)
        else:
            print(f"Creating new {scaler_type} label scaler")
            label_scaler = NutrientScaler(scaler_type=scaler_type)
    
    # Create datasets
    train_dataset = Nutrition5kDataset(
        data_dir=data_dir,
        split_file=os.path.join(split_dir, 'rgb_train_ids.txt'),
        metadata_file=metadata_file,
        transform=train_transform,
        label_scaler=label_scaler,
        fit_scaler=label_scaling and not (scaler_path and os.path.exists(scaler_path))
    )
    
    val_dataset = Nutrition5kDataset(
        data_dir=data_dir,
        split_file=os.path.join(split_dir, 'rgb_test_ids.txt'),
        metadata_file=metadata_file,
        transform=val_transform,
        label_scaler=label_scaler,
        fit_scaler=False
    )
    
    # Save scaler if needed
    if label_scaling and scaler_path and not os.path.exists(scaler_path):
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        print(f"Saving label scaler to {scaler_path}")
        label_scaler.save(scaler_path)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return train_loader, val_loader, label_scaler 
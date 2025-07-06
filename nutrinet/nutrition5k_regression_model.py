import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import glob
import json
from tqdm import tqdm
import time

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class Nutrition5kDataset(Dataset):
    def __init__(self, metadata_file, image_dir, transform=None, normalization='meanstd'):
        """
        Nutrition5k Dataset - RGB Images Only
        
        Args:
            metadata_file: Path to the CSV metadata file
            image_dir: Path to the imagery directory (only rgb.png files used)
            transform: Optional transform to apply to images
            normalization: 'meanstd' (default) or 'max' for normalization method
        """
        self.metadata_file = metadata_file
        self.image_dir = image_dir
        self.transform = transform
        self.normalization = normalization
        self.norm_mean = None
        self.norm_std = None
        self.norm_max = None
        self._norm_ready = False
        
        # Load metadata
        self.load_metadata()
        
        # Compute normalization factors
        self.compute_normalization_factors()
        
    def load_metadata(self):
        """Load and parse the metadata CSV file - Only uses RGB images (rgb.png)"""
        print(f"Loading RGB images from: {self.image_dir}")
        print(f"Reading metadata from: {self.metadata_file}")
        
        # Read the CSV file
        with open(self.metadata_file, 'r') as f:
            lines = f.readlines()
        
        self.data = []
        rgb_images_found = 0
        rgb_images_missing = 0
        
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
                
            dish_id = parts[0]
            
            # Extract nutritional values (first 5 values after dish_id)
            try:
                calories = float(parts[1])
                mass = float(parts[2])
                fat = float(parts[3])
                carbohydrate = float(parts[4])
                protein = float(parts[5])
            except ValueError:
                continue
            
            # Check if RGB image exists (only rgb.png files are used)
            image_path = os.path.join(self.image_dir, dish_id, 'rgb.png')
            if os.path.exists(image_path):
                self.data.append({
                    'dish_id': dish_id,
                    'image_path': image_path,
                    'calories': calories,
                    'mass': mass,
                    'fat': fat,
                    'carbohydrate': carbohydrate,
                    'protein': protein
                })
                rgb_images_found += 1
            else:
                rgb_images_missing += 1
        
        print(f"RGB images found: {rgb_images_found}")
        print(f"RGB images missing: {rgb_images_missing}")
        print(f"Total valid samples: {len(self.data)}")
        print(f"Note: Only rgb.png files are used (depth_color.png and depth_raw.png are ignored)")
    
    def compute_normalization_factors(self):
        """Compute normalization factors (mean/std or max) for the targets."""
        if not hasattr(self, 'data') or len(self.data) == 0:
            return
        targets = np.array([
            [d['calories'], d['mass'], d['fat'], d['carbohydrate'], d['protein']]
            for d in self.data
        ])
        self.norm_mean = targets.mean(axis=0)
        self.norm_std = targets.std(axis=0)
        self.norm_max = targets.max(axis=0)
        self._norm_ready = True

    def normalize_target(self, target):
        if not self._norm_ready:
            return target
        if self.normalization == 'meanstd':
            return (target - self.norm_mean) / (self.norm_std + 1e-8)
        elif self.normalization == 'max':
            return target / (self.norm_max + 1e-8)
        else:
            return target

    def denormalize_target(self, target):
        if not self._norm_ready:
            return target
        if self.normalization == 'meanstd':
            return target * (self.norm_std + 1e-8) + self.norm_mean
        elif self.normalization == 'max':
            return target * (self.norm_max + 1e-8)
        else:
            return target

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load RGB image (convert to RGB to ensure 3-channel format)
        image = Image.open(item['image_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Create target vector (nutritional values)
        target = np.array([
            item['calories'],
            item['mass'],
            item['fat'],
            item['carbohydrate'],
            item['protein']
        ], dtype=np.float32)
        
        norm_target = self.normalize_target(target)
        
        return image, torch.tensor(norm_target, dtype=torch.float32), item['dish_id']  # Return dish_id for specific testing

class NutriNetAlex(nn.Module):
    """
    AlexNet-based architecture for nutrition prediction, inspired by NutriNet
    """
    def __init__(self, num_outputs=5):
        super(NutriNetAlex, self).__init__()
        
        # AlexNet-inspired feature extractor
        self.features = nn.Sequential(
            # Layer 1: Conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 2: Conv2
            nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 3: Conv3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 4: Conv4
            nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            
            # Layer 5: Conv5
            nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Classifier for nutrition prediction
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_outputs)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class FoodClassifier(nn.Module):
    def __init__(self, num_classes=101):
        super(FoodClassifier, self).__init__()
        
        # Use a pre-trained ResNet as feature extractor
        self.features = nn.Sequential(
            # First conv layer
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Residual blocks (simplified)
            self._make_layer(64, 64, 2),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier for food classification
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(self._make_block(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(self._make_block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _make_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def calculate_evaluation_metrics(predictions, targets, norm_factors=None, norm_type=None):
    """
    Calculate comprehensive evaluation metrics for regression
    
    Args:
        predictions: Predicted values (numpy array)
        targets: Actual values (numpy array)
        norm_factors: normalization factors (mean, std, or max)
        norm_type: 'meanstd' or 'max'
    
    Returns:
        Dictionary with MAE, RMSE, R², nMAE, nRMSE, and per-component metrics
    """
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Overall metrics
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    
    # Normalized metrics
    nmae = None
    nrmse = None
    if norm_factors is not None:
        if norm_type == 'meanstd':
            norm = norm_factors['std']
        elif norm_type == 'max':
            norm = norm_factors['max']
        else:
            norm = None
        if norm is not None:
            nmae = mae / (np.mean(norm) + 1e-8)
            nrmse = rmse / (np.mean(norm) + 1e-8)
    
    # Per-component metrics
    component_names = ['Calories', 'Mass', 'Fat', 'Carbohydrate', 'Protein']
    component_metrics = {}
    
    for i, name in enumerate(component_names):
        comp_mae = mean_absolute_error(targets[:, i], predictions[:, i])
        comp_rmse = np.sqrt(mean_squared_error(targets[:, i], predictions[:, i]))
        comp_r2 = r2_score(targets[:, i], predictions[:, i])
        comp_nmae = None
        comp_nrmse = None
        if norm_factors is not None and norm is not None:
            comp_nmae = comp_mae / (norm[i] + 1e-8)
            comp_nrmse = comp_rmse / (norm[i] + 1e-8)
        component_metrics[name] = {
            'MAE': comp_mae,
            'RMSE': comp_rmse,
            'R²': comp_r2,
            'nMAE': comp_nmae,
            'nRMSE': comp_nrmse
        }
    
    return {
        'overall': {
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'nMAE': nmae,
            'nRMSE': nrmse
        },
        'components': component_metrics,
        'norm_factors': norm_factors,
        'norm_type': norm_type
    }

def calculate_accuracy(predictions, targets, tolerance=0.1):
    """
    Calculate accuracy for regression tasks using tolerance-based approach
    """
    # Calculate relative error for each nutritional component
    relative_errors = torch.abs(predictions - targets) / (targets + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Consider prediction accurate if relative error is within tolerance
    accurate_predictions = (relative_errors <= tolerance).float()
    
    # Calculate accuracy as percentage of accurate predictions
    accuracy = accurate_predictions.mean() * 100
    
    return accuracy.item()

def evaluate_model(model, data_loader, device, criterion, norm_factors=None, norm_type=None):
    """
    Evaluate model and return comprehensive metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    
    with torch.no_grad():
        for images, targets, dish_ids in data_loader:
            # Skip batches that are too small for batch normalization
            if images.size(0) < 2:
                continue
                
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            all_predictions.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = calculate_accuracy(all_predictions, all_targets)
    metrics = calculate_evaluation_metrics(all_predictions, all_targets, norm_factors, norm_type)
    
    return avg_loss, accuracy, metrics, all_predictions, all_targets

def test_specific_dishes(model, dataset, device, dish_ids=['dish_1565033189', 'dish_1556572657'], output_file="baseline_result/specific_dish_test_results.txt"):
    """
    Test model on specific dishes, show predicted vs actual values, and save to file.
    Also outputs denormalized (real-world) values if normalization was used.
    """
    print(f"\nTesting specific dishes: {dish_ids}")
    print("=" * 80)
    # Try to get denormalization method
    denorm_fn = getattr(dataset, 'denormalize_target', None)
    dish_data = {}
    for i, (image, target, dish_id) in enumerate(dataset):
        if dish_id in dish_ids:
            dish_data[dish_id] = {'image': image, 'target': target}
    if not dish_data:
        print("None of the specified dishes found in the dataset")
        return
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BASELINE SPECIFIC DISH TEST RESULTS\n")
        f.write("=" * 80 + "\n")
        for dish_id in dish_ids:
            if dish_id not in dish_data:
                result_line = f"\nDish {dish_id} not found in dataset\n"
                print(result_line)
                f.write(result_line)
                continue
            data = dish_data[dish_id]
            image = data['image'].unsqueeze(0).to(device)
            target = data['target']
            model.eval()
            with torch.no_grad():
                prediction = model(image).cpu()[0]
            mae = mean_absolute_error(target.numpy(), prediction.numpy())
            rmse = np.sqrt(mean_squared_error(target.numpy(), prediction.numpy()))
            dish_header = f"\nDish: {dish_id}"
            print(dish_header)
            f.write(dish_header + "\n")
            separator = "-" * 80
            print(separator)
            f.write(separator + "\n")
            header = f"{'Component':<12} | {'Actual':>8} | {'Predicted':>10} | {'Error':>8} | {'Error %':>7}"
            print(header)
            f.write(header + "\n")
            print(separator)
            f.write(separator + "\n")
            component_names = ['Calories', 'Mass (g)', 'Fat (g)', 'Carbs (g)', 'Protein (g)']
            for i, name in enumerate(component_names):
                actual = target[i].item()
                predicted = prediction[i].item()
                error = abs(actual - predicted)
                error_pct = (error / actual) * 100 if actual > 0 else 0
                result_line = f"{name:<12} | {actual:8.2f} | {predicted:10.2f} | {error:8.2f} | {error_pct:6.1f}%"
                print(result_line)
                f.write(result_line + "\n")
            print(separator)
            f.write(separator + "\n")
            metrics_line = f"Overall MAE: {mae:.2f} | Overall RMSE: {rmse:.2f}"
            print(metrics_line)
            f.write(metrics_line + "\n")
            # Denormalized output
            if denorm_fn is not None:
                denorm_actual = denorm_fn(target.numpy())
                denorm_pred = denorm_fn(prediction.numpy())
                f.write("\nDenormalized (real-world) values:\n")
                f.write(f"{'Component':<12} | {'Actual':>12} | {'Predicted':>12} | {'Error':>12} | {'Error %':>9}\n")
                f.write(separator + "\n")
                for i, name in enumerate(component_names):
                    a = denorm_actual[i]
                    p = denorm_pred[i]
                    err = abs(a - p)
                    err_pct = (err / a) * 100 if a > 0 else 0
                    f.write(f"{name:<12} | {a:12.2f} | {p:12.2f} | {err:12.2f} | {err_pct:8.1f}%\n")
                f.write(separator + "\n")
                mae_denorm = mean_absolute_error(denorm_actual, denorm_pred)
                rmse_denorm = np.sqrt(mean_squared_error(denorm_actual, denorm_pred))
                f.write(f"Overall MAE (denorm): {mae_denorm:.2f} | Overall RMSE (denorm): {rmse_denorm:.2f}\n")
    print(f"\nBaseline specific dish test results saved to: {output_file}")

def train_model(model, train_loader, val_loader, num_epochs=15, device='cuda', patience=5, norm_factors=None, norm_type=None):
    """Train the nutrition prediction model with early stopping, progress bars, and checkpoints"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    train_metrics = []
    val_metrics = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs('baseline_checkpoint', exist_ok=True)
    
    print(f"Starting baseline training for {num_epochs} epochs...")
    print(f"Early stopping patience: {patience} epochs")
    print(f"Checkpoints will be saved every 10 epochs to the 'baseline_checkpoint/' directory")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        batch_count = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                         leave=False, ncols=100)
        
        for batch_idx, (images, targets, dish_ids) in enumerate(train_pbar):
            # Skip batches that are too small for batch normalization
            if images.size(0) < 2:
                continue
                
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate accuracy
            acc = calculate_accuracy(outputs, targets)
            
            train_loss += loss.item()
            train_acc += acc
            batch_count += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{acc:.1f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validation phase with comprehensive metrics
        val_loss, val_acc, val_metrics_dict, val_predictions, val_targets = evaluate_model(
            model, val_loader, device, criterion, norm_factors, norm_type
        )
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_metrics.append(calculate_evaluation_metrics(val_predictions, val_targets, norm_factors, norm_type))
        
        # Learning rate scheduling
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr < old_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'baseline_checkpoint/best_baseline_model.pth')
            print(f'New best baseline model saved! Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.1f}%')
        else:
            patience_counter += 1
        
        # Print epoch summary with comprehensive metrics
        print(f'Epoch [{epoch+1}/{num_epochs}] - {epoch_time:.1f}s')
        print(f'   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%')
        print(f'   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}%')
        print(f'   Val MAE: {val_metrics_dict["overall"]["MAE"]:.3f} | Val RMSE: {val_metrics_dict["overall"]["RMSE"]:.3f} | Val R²: {val_metrics_dict["overall"]["R²"]:.3f}')
        print(f'   LR: {optimizer.param_groups[0]["lr"]:.6f} | Patience: {patience_counter}/{patience}')
        print('-' * 60)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_metrics': val_metrics_dict,
                'best_val_loss': best_val_loss
            }
            checkpoint_path = f'baseline_checkpoint/checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f'Baseline checkpoint saved: {checkpoint_path}')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs!')
            break
    
    # Load best model
    model.load_state_dict(torch.load('baseline_checkpoint/best_baseline_model.pth'))
    print(f'Baseline training completed! Best validation loss: {best_val_loss:.4f}')
    print(f'Best validation accuracy: {max(val_accuracies):.1f}%')
    
    # Final evaluation with best model
    print("\nFinal Baseline Model Evaluation:")
    print("=" * 60)
    final_val_loss, final_val_acc, final_val_metrics, final_val_predictions, final_val_targets = evaluate_model(
        model, val_loader, device, criterion, norm_factors, norm_type
    )
    
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Final Validation Accuracy: {final_val_acc:.1f}%")
    print(f"Final Validation MAE: {final_val_metrics['overall']['MAE']:.3f}")
    print(f"Final Validation RMSE: {final_val_metrics['overall']['RMSE']:.3f}")
    print(f"Final Validation R²: {final_val_metrics['overall']['R²']:.3f}")
    
    # Per-component metrics
    print("\nPer-Component Metrics:")
    print("-" * 60)
    for component, metrics in final_val_metrics['components'].items():
        print(f"{component:12} | MAE: {metrics['MAE']:6.3f} | RMSE: {metrics['RMSE']:6.3f} | R²: {metrics['R²']:6.3f}")
    
    return train_losses, val_losses, train_accuracies, val_accuracies, train_metrics

def predict_nutrition_from_image(image_path, model, device='cuda'):
    """
    Predict nutritional values from an uploaded RGB image
    
    Args:
        image_path: Path to the uploaded RGB image
        model: Trained nutrition prediction model
        device: Device to run inference on
    
    Returns:
        Dictionary with predicted nutritional values
    """
    # Define transforms for RGB images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess RGB image
    try:
        image = Image.open(image_path).convert('RGB')  # Ensure RGB format
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        return {"error": f"Failed to load RGB image: {str(e)}"}
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)
        prediction = prediction.cpu().numpy()[0]
    
    # Return results
    return {
        "calories": float(prediction[0]),
        "mass_grams": float(prediction[1]),
        "fat_grams": float(prediction[2]),
        "carbohydrates_grams": float(prediction[3]),
        "protein_grams": float(prediction[4])
    }

def classify_food_and_predict_nutrition(image_path, nutrition_model, food_classifier=None, device='cuda'):
    """
    Classify food and predict nutritional values from an uploaded image
    
    Args:
        image_path: Path to the uploaded image
        nutrition_model: Trained nutrition prediction model
        food_classifier: Optional food classification model
        device: Device to run inference on
    
    Returns:
        Dictionary with food classification and nutritional values
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        return {"error": f"Failed to load image: {str(e)}"}
    
    # Make nutrition prediction
    nutrition_model.eval()
    with torch.no_grad():
        nutrition_prediction = nutrition_model(image_tensor)
        nutrition_prediction = nutrition_prediction.cpu().numpy()[0]
    
    # Simple food classification based on visual features
    # In a real implementation, you would use a trained food classifier
    food_name = "Food Item"  # Placeholder
    
    # Return results
    return {
        "food_name": food_name,
        "calories": float(nutrition_prediction[0]),
        "mass_grams": float(nutrition_prediction[1]),
        "fat_grams": float(nutrition_prediction[2]),
        "carbohydrates_grams": float(nutrition_prediction[3]),
        "protein_grams": float(nutrition_prediction[4]),
        "estimated_serving_size": f"{nutrition_prediction[1]:.1f}g"
    }

def plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies, val_metrics=None):
    """Plot training and validation losses and accuracies"""
    
    # Create a single figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Baseline Training Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Losses
    ax1.plot(train_losses, label='Train Loss', color='blue', linewidth=2)
    ax1.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Accuracies
    ax2.plot(train_accuracies, label='Train Accuracy', color='blue', linewidth=2)
    ax2.plot(val_accuracies, label='Validation Accuracy', color='red', linewidth=2)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('baseline_result/baseline_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save metrics summary
    if val_metrics:
        save_metrics_summary(val_metrics)

def save_metrics_summary(val_metrics):
    """Save a summary of the final metrics to a text file, including normalized metrics and normalization factors."""
    if not val_metrics:
        return
    final_metrics = val_metrics[-1]
    with open('baseline_result/baseline_training_metrics_summary.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("BASELINE TRAINING METRICS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write("OVERALL METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"MAE:  {final_metrics['overall']['MAE']:.4f}\n")
        f.write(f"RMSE: {final_metrics['overall']['RMSE']:.4f}\n")
        f.write(f"R²:   {final_metrics['overall']['R²']:.4f}\n")
        if final_metrics['overall'].get('nMAE') is not None:
            f.write(f"nMAE: {final_metrics['overall']['nMAE']:.4f}\n")
        if final_metrics['overall'].get('nRMSE') is not None:
            f.write(f"nRMSE: {final_metrics['overall']['nRMSE']:.4f}\n")
        f.write("\n")
        f.write("PER-COMPONENT METRICS:\n")
        f.write("-" * 30 + "\n")
        for component, metrics in final_metrics['components'].items():
            f.write(f"{component}:\n")
            f.write(f"  MAE:  {metrics['MAE']:.4f}\n")
            f.write(f"  RMSE: {metrics['RMSE']:.4f}\n")
            f.write(f"  R²:   {metrics['R²']:.4f}\n")
            if metrics.get('nMAE') is not None:
                f.write(f"  nMAE:  {metrics['nMAE']:.4f}\n")
            if metrics.get('nRMSE') is not None:
                f.write(f"  nRMSE: {metrics['nRMSE']:.4f}\n")
            f.write("\n")
        # Save normalization factors
        if final_metrics.get('norm_factors') is not None:
            f.write("Normalization factors used (mean/std or max):\n")
            for k, v in final_metrics['norm_factors'].items():
                f.write(f"  {k}: {v}\n")
            f.write(f"Normalization type: {final_metrics.get('norm_type')}\n")
    print("Baseline metrics summary saved to: baseline_result/baseline_training_metrics_summary.txt")

def save_evaluation_metrics(metrics, output_file, predictions=None, targets=None, denorm_fn=None):
    """Save overall and per-component evaluation metrics to a text file. Optionally outputs denormalized predictions/targets."""
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("FINAL BASELINE EVALUATION METRICS\n")
        f.write("=" * 60 + "\n\n")
        f.write("OVERALL METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"MAE:  {metrics['overall']['MAE']:.4f}\n")
        f.write(f"RMSE: {metrics['overall']['RMSE']:.4f}\n")
        f.write(f"R²:   {metrics['overall']['R²']:.4f}\n\n")
        f.write("PER-COMPONENT METRICS:\n")
        f.write("-" * 30 + "\n")
        for component, m in metrics['components'].items():
            f.write(f"{component}:\n")
            f.write(f"  MAE:  {m['MAE']:.4f}\n")
            f.write(f"  RMSE: {m['RMSE']:.4f}\n")
            f.write(f"  R²:   {m['R²']:.4f}\n\n")
        # Optionally output denormalized predictions/targets
        if predictions is not None and targets is not None and denorm_fn is not None:
            f.write("\nSample of denormalized predictions and targets (first 10):\n")
            component_names = ['Calories', 'Mass (g)', 'Fat (g)', 'Carbs (g)', 'Protein (g)']
            f.write(f"{'Idx':<4} | " + " | ".join([f'{n:<12}' for n in component_names]) + "\n")
            f.write("-" * 80 + "\n")
            for i in range(min(10, len(predictions))):
                denorm_pred = denorm_fn(predictions[i])
                denorm_tgt = denorm_fn(targets[i])
                f.write(f"{i:<4} | " + " | ".join([f"{denorm_tgt[j]:10.2f}/{denorm_pred[j]:10.2f}" for j in range(5)]) + "\n")
            f.write("(Format: target/prediction)\n")
    print(f"Final baseline evaluation metrics saved to: {output_file}")

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data paths
    metadata_file = 'nutrition5k_dataset/metadata/dish_metadata_cafe1.csv'
    image_dir = 'nutrition5k_dataset/imagery/realsense_overhead'
    
    print("=" * 60)
    print("Baseline Food Nutrition Prediction - RGB Images Only")
    print("=" * 60)
    print(f"Image directory: {image_dir}")
    print(f"Metadata file: {metadata_file}")
    print("Only rgb.png files will be used for training")
    print("   (depth_color.png and depth_raw.png are ignored)")
    print("Using AlexNet-inspired architecture with early stopping")
    print("=" * 60)
    
    # Check if data exists
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file not found at {metadata_file}")
        return
    
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found at {image_dir}")
        return
    
    # Data transforms for RGB images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset (RGB images only) with normalization
    dataset = Nutrition5kDataset(metadata_file, image_dir, transform=transform, normalization='meanstd')
    
    if len(dataset) == 0:
        print("Error: No valid data found in dataset")
        return
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Calculate batch size based on dataset size
    batch_size = min(16, max(4, len(dataset) // 20))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    
    # Set fixed number of epochs for baseline
    num_epochs = 15
    patience = 5
    
    print(f"Training for {num_epochs} epochs")
    print(f"Early stopping patience: {patience}")
    
    # Initialize model
    model = NutriNetAlex(num_outputs=5).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Pass normalization factors to metric calculation
    norm_factors = {'mean': dataset.norm_mean, 'std': dataset.norm_std, 'max': dataset.norm_max}
    norm_type = dataset.normalization
    
    # Train model
    print("Starting baseline training...")
    train_losses, val_losses, train_accuracies, val_accuracies, train_metrics = train_model(
        model, train_loader, val_loader, num_epochs=num_epochs, device=device, patience=patience,
        norm_factors=norm_factors, norm_type=norm_type
    )
    
    # Always load the best model after training (in case of early stopping)
    model.load_state_dict(torch.load('baseline_checkpoint/best_baseline_model.pth'))
    print('[DEBUG] Baseline training complete. Best model loaded. Proceeding to evaluation and specific dish test.')
    
    # Save the model to model.pth
    torch.save(model.state_dict(), 'model.pth')
    print('Model saved to model.pth')
    
    # Final evaluation with best model
    print("[DEBUG] About to run final evaluation block...")
    criterion = nn.MSELoss()
    print("\nFinal Baseline Model Evaluation:")
    print("=" * 60)
    final_val_loss, final_val_acc, final_val_metrics, final_val_predictions, final_val_targets = evaluate_model(
        model, val_loader, device, criterion, norm_factors, norm_type
    )
    print("[DEBUG] Finished running final evaluation block.")
    
    # Save evaluation metrics to file
    os.makedirs('baseline_result', exist_ok=True)
    save_evaluation_metrics(final_val_metrics, 'baseline_result/evaluation_metrics.txt', predictions=final_val_predictions, targets=final_val_targets, denorm_fn=dataset.denormalize_target)
    
    # Save metrics summary
    save_metrics_summary(train_metrics)
    
    # Plot training results
    plot_training_results(train_losses, val_losses, train_accuracies, val_accuracies, train_metrics)
    
    # Test specific dishes and save results to file
    # Always use the full dataset for specific dish testing to ensure all dish_ids with images are available
    test_specific_dishes(model, dataset, device, output_file="baseline_result/specific_dish_test_results.txt")

if __name__ == "__main__":
    main() 
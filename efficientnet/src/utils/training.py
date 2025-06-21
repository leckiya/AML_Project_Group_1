import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm


class NutrientLoss(nn.Module):
    """Custom loss function for nutrient prediction"""
    def __init__(self, weights=None):
        """
        Args:
            weights (list, optional): Custom weights for each nutrient
        """
        super(NutrientLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
        
        if weights is None:
            self.weights = torch.tensor([0.1, 0.3, 0.2, 0.2, 0.2])
        else:
            self.weights = torch.tensor(weights)
            
    def forward(self, predictions, targets):
        """
        Compute weighted MSE loss
        
        Args:
            predictions (torch.Tensor): Predicted nutrient values [B, 5]
            targets (torch.Tensor): Ground truth nutrient values [B, 5]
            
        Returns:
            torch.Tensor: Loss value
        """
        element_wise_loss = self.mse(predictions, targets)
        weighted_loss = element_wise_loss * self.weights.to(element_wise_loss.device)
        
        sample_loss = torch.sum(weighted_loss, dim=1)
        
        return torch.mean(sample_loss)


def train_epoch(model, dataloader, criterion, optimizer, device, log_interval=10):
    """
    Train model for one epoch
    
    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to use for training
        log_interval (int): Interval for logging
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    
    for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc='Training')):
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        total_loss += loss.item()
        
        if (batch_idx + 1) % log_interval == 0:
            print(f'Batch {batch_idx+1}/{len(dataloader)}, Loss: {running_loss/log_interval:.4f}')
            running_loss = 0.0
            
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Validate model on validation set
    
    Args:
        model (nn.Module): Model to validate
        dataloader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to use
        
    Returns:
        tuple: (average loss, predictions, targets)
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    return total_loss / len(dataloader), all_predictions, all_targets


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    learning_rate=0.001,
    weight_decay=1e-5,
    scheduler_patience=5,
    scheduler_factor=0.5,
    log_dir='logs',
    checkpoint_dir='checkpoints'
):
    """
    Train model for multiple epochs
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of training epochs
        learning_rate (float): Initial learning rate
        weight_decay (float): Weight decay factor
        scheduler_patience (int): Patience for learning rate scheduler
        scheduler_factor (float): Factor for learning rate reduction
        log_dir (str): Directory for TensorBoard logs
        checkpoint_dir (str): Directory for model checkpoints
        
    Returns:
        nn.Module: Trained model
    """
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    writer = SummaryWriter(log_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for training.")
    device = torch.device("cuda")
    model = model.to(device)
    
    criterion = NutrientLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=scheduler_factor,
        patience=scheduler_patience
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        val_loss, val_preds, val_targets = validate(model, val_loader, criterion, device)
        
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            print(f"Learning rate changed from {old_lr} to {new_lr}")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
            print("Saved best model checkpoint")
            
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth'))
        
        time_taken = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {time_taken:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
    writer.close()
    return model 
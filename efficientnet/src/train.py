import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import get_data_loaders
from src.models.efficientnet_model import get_model, get_v2_model
from src.utils.training import train_model
from src.evaluation.metrics import compute_metrics, print_metrics, plot_prediction_vs_target


def parse_args():
    parser = argparse.ArgumentParser(description='Train nutrient prediction model')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='nutrition5k_dataset/imagery',
                        help='Path to imagery directory')
    parser.add_argument('--split_dir', type=str, default='nutrition5k_dataset/dish_ids/splits',
                        help='Path to split directory')
    parser.add_argument('--metadata_file', type=str, default='nutrition5k_dataset/metadata/dish_metadata_cafe1.csv',
                        help='Path to metadata file')
    
    # Model parameters
    parser.add_argument('--efficientnet_version', type=str, default='b0',
                        help='EfficientNet version (b0, b1, b2, etc.)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained weights')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=32,
                        help='Number of data loader workers')
    
    # Label scaling parameters
    parser.add_argument('--label_scaling', action='store_true',
                        help='Enable label scaling')
    parser.add_argument('--scaler_type', type=str, default='standard', choices=['standard', 'minmax'],
                        help='Type of label scaler (standard or minmax)')
    
    # Output parameters
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory for model checkpoints')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory for evaluation results')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for training.")
    device = torch.device("cuda")
    print("Using device: cuda")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name = f"efficientnet_{args.efficientnet_version}"
    if args.label_scaling:
        model_name += f"_scaled_{args.scaler_type}"
    model_name += f"_{timestamp}"
    
    log_dir = os.path.join(args.log_dir, model_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, model_name)
    results_dir = os.path.join(args.results_dir, model_name)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    scaler_path = None
    if args.label_scaling:
        scaler_path = os.path.join(checkpoint_dir, 'label_scaler.pkl')
    
    print("Loading data...")
    train_loader, val_loader, label_scaler = get_data_loaders(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        metadata_file=args.metadata_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_scaling=args.label_scaling,
        scaler_type=args.scaler_type,
        scaler_path=scaler_path
    )
    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    
    model_metadata = {
        'timestamp': timestamp,
        'label_scaling': args.label_scaling,
        'scaler_type': args.scaler_type if args.label_scaling else None
    }
    
    with open(os.path.join(checkpoint_dir, 'model_metadata.txt'), 'w') as f:
        for key, value in model_metadata.items():
            f.write(f"{key}: {value}\n")
    
    print("Creating model...")
    if args.efficientnet_version.startswith('v2_'):
        # Use EfficientNetV2 model
        model = get_v2_model(
            efficientnet_version=args.efficientnet_version[3:],  # remove 'v2_' prefix
            num_outputs=5,
            pretrained=args.pretrained
        )
    else:
        # Use original EfficientNet model
        model = get_model(
            efficientnet_version=args.efficientnet_version,
            num_outputs=5,  # mass, calories, fat, carbs, protein
            pretrained=args.pretrained
        )
    
    print("Starting training...")
    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir
    )
    
    best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Make sure the model is on the correct device
    model = model.to(device)
    print(f"Loaded best model from epoch {checkpoint['epoch']} with validation loss {checkpoint['val_loss']:.4f}")
    
    print("Evaluating model...")
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # If label scaling was applied, inverse transform the predictions and targets back to original scale
    if args.label_scaling and label_scaler is not None:
        print("Inverse transforming scaled predictions and targets for evaluation")
        all_predictions = label_scaler.inverse_transform(all_predictions)
        all_targets = label_scaler.inverse_transform(all_targets)
    
    output_names = ['mass', 'calories', 'fat', 'carbs', 'protein']
    metrics = compute_metrics(all_predictions, all_targets, output_names)
    
    print_metrics(metrics)
    
    metrics_path = os.path.join(results_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("===== Training Configuration =====\n")
        f.write(f"Label scaling: {args.label_scaling}\n")
        if args.label_scaling:
            f.write(f"Scaler type: {args.scaler_type}\n")
            if label_scaler.scaler_type == 'standard':
                f.write(f"Mean: {label_scaler.scaler.mean_}\n")
                f.write(f"Scale: {label_scaler.scaler.scale_}\n")
            else:
                f.write(f"Data min: {label_scaler.scaler.data_min_}\n")
                f.write(f"Data max: {label_scaler.scaler.data_max_}\n")
        
        f.write("\n===== Overall Metrics =====\n")
        for metric, value in metrics['overall'].items():
            f.write(f"{metric}: {value}\n")
        
        f.write("\n===== Per-Output Metrics =====\n")
        for output_name, output_metrics in metrics['per_output'].items():
            f.write(f"\n-- {output_name.upper()} --\n")
            for metric, value in output_metrics.items():
                f.write(f"{metric}: {value}\n")
    
    plt.figure(figsize=(15, 10))
    prediction_fig = plot_prediction_vs_target(
        predictions=all_predictions,
        targets=all_targets,
        output_names=output_names,
        save_path=os.path.join(results_dir, 'prediction_vs_target.png')
    )
    
    print(f"Results saved to {results_dir}")
    print(f"Training completed. Best model saved to {best_checkpoint_path}")
    if args.label_scaling:
        print(f"Label scaler saved to {scaler_path}")


if __name__ == "__main__":
    main() 
import os
import argparse
import torch
import numpy as np
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import get_data_loaders
from src.models.efficientnet_model import get_model, get_v2_model
from src.evaluation.metrics import compute_metrics, print_metrics, plot_prediction_vs_target
from src.utils.scaler import NutrientScaler
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate nutrient prediction model')
    
    parser.add_argument('--data_dir', type=str, default='nutrition5k_dataset/imagery',
                        help='Path to imagery directory')
    parser.add_argument('--split_dir', type=str, default='nutrition5k_dataset/dish_ids/splits',
                        help='Path to split directory')
    parser.add_argument('--metadata_file', type=str, default='nutrition5k_dataset/metadata/dish_metadata_cafe1.csv',
                        help='Path to metadata file')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--efficientnet_version', type=str, default='b0',
                        help='EfficientNet version (b0, b1, b2, etc.)')
    parser.add_argument('--is_v2', action='store_true',
                        help='Use EfficientNetV2 model')
    
    parser.add_argument('--label_scaling', action='store_true',
                        help='Whether the model was trained with label scaling')
    parser.add_argument('--scaler_path', type=str, default=None,
                        help='Path to the label scaler file')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--results_dir', type=str, default='results/evaluation',
                        help='Directory for evaluation results')
    parser.add_argument('--avg_r2', action='store_true',
                        help='Use averaged R² instead of flattened approach')
    parser.add_argument('--normalization', type=str, default='range', choices=['range', 'mean', 'std'],
                        help='Normalization method for metrics')
    
    return parser.parse_args()


def calculate_averaged_r2(targets, predictions):
    """
    Calculate R² by averaging individual R² for each nutrient
    
    Args:
        targets (np.ndarray): Ground truth values [N, num_outputs]
        predictions (np.ndarray): Predicted values [N, num_outputs]
        
    Returns:
        float: Averaged R² value
    """
    from sklearn.metrics import r2_score
    
    num_outputs = targets.shape[1]
    r2_values = []
    
    for i in range(num_outputs):
        r2 = r2_score(targets[:, i], predictions[:, i])
        r2_values.append(r2)
        
    return np.mean(r2_values)


def evaluate_model(model, val_loader, label_scaler=None, avg_r2=False, normalization='range'):
    """
    Evaluate model on validation data
    
    Args:
        model (nn.Module): Model to evaluate
        val_loader (DataLoader): Validation data loader
        label_scaler (NutrientScaler, optional): Scaler for labels
        avg_r2 (bool): Use averaged R² instead of flattened
        normalization (str): Normalization method for metrics ('range', 'mean', or 'std')
        
    Returns:
        tuple: (all_predictions, all_targets, metrics)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("Running inference on validation set...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_predictions = np.vstack(all_predictions)
    all_targets = np.vstack(all_targets)
    
    # If label scaling was applied, inverse transform the predictions and targets
    if label_scaler is not None:
        print("Inverse transforming scaled predictions and targets for evaluation")
        all_predictions = label_scaler.inverse_transform(all_predictions)
        all_targets = label_scaler.inverse_transform(all_targets)
    
    output_names = ['mass', 'calories', 'fat', 'carbs', 'protein']
    metrics = compute_metrics(all_predictions, all_targets, output_names, normalization=normalization)
    
    if avg_r2:
        avg_r2_value = calculate_averaged_r2(all_targets, all_predictions)
        metrics['overall']['avg_r2'] = avg_r2_value
        print(f"\nAveraged R² (mean of individual R² values): {avg_r2_value:.4f}")
    
    return all_predictions, all_targets, metrics


def main():
    args = parse_args()
    
    if not torch.cuda.is_available():
        print("Warning: CUDA is not available. Using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"Using device: {device}")
    
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Try to determine if the model used label scaling from metadata file
    if not args.label_scaling and not args.scaler_path:
        model_dir = os.path.dirname(args.model_path)
        metadata_path = os.path.join(model_dir, 'model_metadata.txt')
        scaler_path = os.path.join(model_dir, 'label_scaler.pkl')
        
        if os.path.exists(metadata_path):
            print(f"Found model metadata file: {metadata_path}")
            with open(metadata_path, 'r') as f:
                metadata_lines = f.readlines()
            
            for line in metadata_lines:
                if 'label_scaling: True' in line:
                    args.label_scaling = True
                    print("Detected that model was trained with label scaling")
                    
                    if os.path.exists(scaler_path) and not args.scaler_path:
                        args.scaler_path = scaler_path
                        print(f"Using scaler from model directory: {args.scaler_path}")
    
    label_scaler = None
    if args.label_scaling:
        if not args.scaler_path or not os.path.exists(args.scaler_path):
            print("Warning: Label scaling is enabled but no valid scaler path provided. Looking in model directory...")
            model_dir = os.path.dirname(args.model_path)
            scaler_path = os.path.join(model_dir, 'label_scaler.pkl')
            if os.path.exists(scaler_path):
                args.scaler_path = scaler_path
                print(f"Found scaler at: {args.scaler_path}")
            else:
                raise ValueError("Label scaling is enabled but no valid scaler path found")
        
        print(f"Loading label scaler from {args.scaler_path}")
        label_scaler = NutrientScaler.load(args.scaler_path)
        print(f"Loaded {label_scaler.scaler_type} scaler")
        
        if label_scaler.scaler_type == 'standard':
            print(f"Mean: {label_scaler.scaler.mean_}")
            print(f"Scale: {label_scaler.scaler.scale_}")
        else:
            print(f"Data min: {label_scaler.scaler.data_min_}")
            print(f"Data max: {label_scaler.scaler.data_max_}")
    
    print("Loading data...")
    _, val_loader, _ = get_data_loaders(
        data_dir=args.data_dir,
        split_dir=args.split_dir,
        metadata_file=args.metadata_file,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        label_scaling=args.label_scaling,
        scaler_path=args.scaler_path if args.label_scaling else None
    )
    print(f"Validation set size: {len(val_loader.dataset)}")
    
    print("Loading model...")
    if args.is_v2:
        model = get_v2_model(
            efficientnet_version=args.efficientnet_version,
            num_outputs=5,
            pretrained=False
        )
    else:
        model = get_model(
            efficientnet_version=args.efficientnet_version,
            num_outputs=5,
            pretrained=False
        )
    
    print(f"Loading checkpoint from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')} with validation loss {checkpoint.get('val_loss', 'unknown')}")
    
    print("Evaluating model...")
    all_predictions, all_targets, metrics = evaluate_model(
        model, 
        val_loader,
        label_scaler=label_scaler,
        avg_r2=args.avg_r2,
        normalization=args.normalization
    )
    
    print_metrics(metrics)
    
    metrics_path = os.path.join(args.results_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("===== Evaluation Configuration =====\n")
        f.write(f"Model path: {args.model_path}\n")
        f.write(f"Label scaling: {args.label_scaling}\n")
        if args.label_scaling and label_scaler:
            f.write(f"Scaler type: {label_scaler.scaler_type}\n")
            if label_scaler.scaler_type == 'standard':
                f.write(f"Mean: {label_scaler.scaler.mean_}\n")
                f.write(f"Scale: {label_scaler.scaler.scale_}\n")
            else:
                f.write(f"Data min: {label_scaler.scaler.data_min_}\n")
                f.write(f"Data max: {label_scaler.scaler.data_max_}\n")
        f.write(f"Metric normalization: {args.normalization}\n\n")
        
        f.write(f"===== Overall Metrics (Normalization: {args.normalization}) =====\n")
        for metric, value in metrics['overall'].items():
            f.write(f"{metric}: {value}\n")
        
        f.write("\n===== Per-Output Metrics =====\n")
        for output_name, output_metrics in metrics['per_output'].items():
            f.write(f"\n-- {output_name.upper()} --\n")
            for metric, value in output_metrics.items():
                f.write(f"{metric}: {value}\n")
    
    output_names = ['mass', 'calories', 'fat', 'carbs', 'protein']
    plt.figure(figsize=(15, 10))
    plot_path = os.path.join(args.results_dir, 'prediction_vs_target.png')
    prediction_fig = plot_prediction_vs_target(
        predictions=all_predictions,
        targets=all_targets,
        output_names=output_names,
        save_path=plot_path,
        normalization=args.normalization
    )
    
    print(f"Results saved to {args.results_dir}")


if __name__ == "__main__":
    main() 
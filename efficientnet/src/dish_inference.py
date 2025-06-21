import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.efficientnet_model import get_model, get_v2_model
from src.utils.scaler import NutrientScaler

def load_metadata(metadata_file):
    """
    Load metadata from file
    
    Args:
        metadata_file (str): Path to metadata file
    
    Returns:
        dict: Mapping from dish ID to nutrient values
    """
    metadata = {}
    
    # Check if the metadata file is in the processed format (with headers)
    with open(metadata_file, 'r') as f:
        first_line = f.readline().strip()
        has_header = 'dish_id,mass,calories' in first_line
    
    if has_header:
        # Processed metadata with headers
        import pandas as pd
        df = pd.read_csv(metadata_file)
        
        for _, row in df.iterrows():
            dish_id = row['dish_id']
            metadata[dish_id] = {
                'calories': float(row['calories']),
                'mass': float(row['mass']),
                'fat': float(row['fat']),
                'carbs': float(row['carbs']),
                'protein': float(row['protein'])
            }
    else:
        # Original format
        with open(metadata_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                    
                dish_id = parts[0].strip()
                try:
                    metadata[dish_id] = {
                        'mass': float(parts[1]),
                        'calories': float(parts[2]),
                        'fat': float(parts[3]),
                        'carbs': float(parts[4]),
                        'protein': float(parts[5])
                    }
                except (ValueError, IndexError):
                    continue
    
    return metadata

def predict_dish(model, dish_id, data_dir, label_scaler=None):
    """
    Run inference on a specific dish
    
    Args:
        model (nn.Module): Trained model
        dish_id (str): Dish ID
        data_dir (str): Path to imagery directory
        label_scaler (NutrientScaler, optional): Scaler for labels
        
    Returns:
        dict: Predicted nutrient values
    """
    # Only CUDA is supported
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_path = os.path.join(data_dir, 'realsense_overhead', dish_id, 'rgb.png')
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)
        
        # Inverse transform if using label scaling
        if label_scaler is not None:
            predictions = label_scaler.inverse_transform(predictions)
    
    predictions = predictions.cpu().numpy()[0]
    
    return {
        'calories': predictions[1],
        'mass': predictions[0],
        'fat': predictions[2],
        'carbs': predictions[3],
        'protein': predictions[4]
    }

def display_results(dish_id, actual_values, predicted_values, normalization='range'):
    """
    Display results in a formatted table
    
    Args:
        dish_id (str): Dish ID
        actual_values (dict): Ground truth nutrient values
        predicted_values (dict): Predicted nutrient values
        normalization (str): Method for normalizing errors ('range', 'mean', or 'std')
    """
    components = ['calories', 'mass', 'fat', 'carbs', 'protein']
    
    errors = {}
    error_percentages = {}
    abs_errors = []
    norm_factors = {}
    norm_errors = {}
    
    for comp in components:
        actual = actual_values[comp]
        predicted = predicted_values[comp]
        errors[comp] = predicted - actual
        
        if actual != 0:
            error_percentages[comp] = abs(errors[comp] / actual) * 100
        else:
            error_percentages[comp] = float('inf')
            
        abs_errors.append(abs(errors[comp]))
        
        # Calculate normalization factor based on the selected method
        if normalization == 'range':
            # We only have one value, so use a typical range for each component
            ranges = {
                'calories': 2000,  # Typical range for calories: 0-2000
                'mass': 1000,      # Typical range for mass: 0-1000g
                'fat': 100,        # Typical range for fat: 0-100g
                'carbs': 300,      # Typical range for carbs: 0-300g
                'protein': 100     # Typical range for protein: 0-100g
            }
            norm_factor = ranges[comp]
        elif normalization == 'mean':
            norm_factor = actual if actual != 0 else 1.0
        elif normalization == 'std':
            # Without a dataset, we use a typical std dev as a percentage of the value
            # This is just an approximation for display purposes
            typical_std_percent = {
                'calories': 0.2,
                'mass': 0.2,
                'fat': 0.3,
                'carbs': 0.3,
                'protein': 0.3
            }
            norm_factor = actual * typical_std_percent[comp] if actual != 0 else 1.0
        else:
            norm_factor = 1.0  # Default: no normalization
            
        norm_factors[comp] = norm_factor
        norm_errors[comp] = abs(errors[comp]) / norm_factor if norm_factor != 0 else float('inf')
    
    overall_mae = np.mean(abs_errors)
    overall_rmse = np.sqrt(np.mean(np.square(abs_errors)))
    overall_nmae = np.mean([norm_errors[comp] for comp in components])
    overall_nrmse = np.sqrt(np.mean([norm_errors[comp]**2 for comp in components]))
    
    print("=" * 80)
    print("NUTRIENT SPECIFIC DISH TEST RESULTS")
    print("=" * 80)
    print(f"Dish: {dish_id}")
    print("-" * 80)
    print(f"{'Component':<12} | {'Actual':>8} | {'Predicted':>10} | {'Error':>6} | {'Error %':>7} | {'Norm Err':>8}")
    print("-" * 80)
    
    for comp in components:
        if comp == "calories":
            print(f"{'Calories':<12} | {actual_values[comp]:>8.2f} | {predicted_values[comp]:>10.2f} | {errors[comp]:>6.2f} | {error_percentages[comp]:>6.1f}% | {norm_errors[comp]:>8.4f}")
        elif comp == "mass":
            print(f"{'Mass (g)':<12} | {actual_values[comp]:>8.2f} | {predicted_values[comp]:>10.2f} | {errors[comp]:>6.2f} | {error_percentages[comp]:>6.1f}% | {norm_errors[comp]:>8.4f}")
        elif comp == "fat":
            print(f"{'Fat (g)':<12} | {actual_values[comp]:>8.2f} | {predicted_values[comp]:>10.2f} | {errors[comp]:>6.2f} | {error_percentages[comp]:>6.1f}% | {norm_errors[comp]:>8.4f}")
        elif comp == "carbs":
            print(f"{'Carbs (g)':<12} | {actual_values[comp]:>8.2f} | {predicted_values[comp]:>10.2f} | {errors[comp]:>6.2f} | {error_percentages[comp]:>6.1f}% | {norm_errors[comp]:>8.4f}")
        elif comp == "protein":
            print(f"{'Protein (g)':<12} | {actual_values[comp]:>8.2f} | {predicted_values[comp]:>10.2f} | {errors[comp]:>6.2f} | {error_percentages[comp]:>6.1f}% | {norm_errors[comp]:>8.4f}")
    
    print("-" * 80)
    print(f"Overall MAE: {overall_mae:.2f} | Overall RMSE: {overall_rmse:.2f}")
    print(f"Overall NMAE ({normalization}): {overall_nmae:.4f} | Overall NRMSE ({normalization}): {overall_nrmse:.4f}")
    
    return abs_errors, norm_errors, norm_factors

def main():
    parser = argparse.ArgumentParser(description='Run inference on specific dishes')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='nutrition5k_dataset/imagery', help='Path to imagery directory')
    parser.add_argument('--metadata_file', type=str, default='nutrition5k_dataset/metadata/dish_metadata_cafe1.csv', help='Path to metadata file')
    parser.add_argument('--model_version', type=str, default='b0', help='EfficientNet version (b0, b1, etc.)')
    parser.add_argument('--is_v2', action='store_true', help='Use EfficientNetV2 instead of original')
    parser.add_argument('--dish_ids', type=str, nargs='+', help='Specific dish IDs to process (e.g., dish_1565033189)')
    parser.add_argument('--all_dishes', action='store_true', help='Process all dishes in the metadata file')
    parser.add_argument('--max_dishes', type=int, default=None, help='Maximum number of dishes to process when using --all_dishes')
    parser.add_argument('--normalization', type=str, default='range', choices=['range', 'mean', 'std'],
                        help='Normalization method for error metrics')
    parser.add_argument('--label_scaling', action='store_true', help='Whether the model was trained with label scaling')
    parser.add_argument('--scaler_path', type=str, default=None, help='Path to label scaler file')
    
    args = parser.parse_args()
    
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
    
    # Load label scaler if specified
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
    
    # Determine which dish IDs to test
    if args.all_dishes:
        # Load metadata
        metadata = load_metadata(args.metadata_file)
        dish_ids = list(metadata.keys())
        
        if args.max_dishes:
            dish_ids = dish_ids[:args.max_dishes]
            
        print(f"Testing on {len(dish_ids)} dishes from metadata")
    elif args.dish_ids:
        dish_ids = args.dish_ids
        print(f"Testing on {len(dish_ids)} specified dishes")
    else:
        # Default dish IDs if none specified
        dish_ids = ['dish_1565033189', 'dish_1556572657']
        print(f"Testing on {len(dish_ids)} default dishes")
    
    # Load metadata
    metadata = load_metadata(args.metadata_file)
    
    # Load model
    if args.is_v2:
        model = get_v2_model(
            efficientnet_version=args.model_version,
            num_outputs=5,
            pretrained=False
        )
    else:
        model = get_model(
            efficientnet_version=args.model_version,
            num_outputs=5,
            pretrained=False
        )
    
    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)  # Ensure the model is on the correct device
    model.eval()
    
    # Lists to collect all errors
    all_abs_errors = []
    all_norm_errors = {}
    norm_factors = {}
    component_errors = {
        'calories': [],
        'mass': [],
        'fat': [],
        'carbs': [],
        'protein': []
    }
    component_norm_errors = {
        'calories': [],
        'mass': [],
        'fat': [],
        'carbs': [],
        'protein': []
    }
    processed_count = 0
    failed_count = 0
    
    # Process each dish
    for dish_id in dish_ids:
        if dish_id not in metadata:
            print(f"Warning: No metadata found for {dish_id}")
            continue
            
        try:
            # Get actual values
            actual_values = metadata[dish_id]
            
            # Run inference
            predicted_values = predict_dish(model, dish_id, args.data_dir, label_scaler=label_scaler)
            
            # Display results
            abs_errors, norm_errors, dish_norm_factors = display_results(
                dish_id, 
                actual_values, 
                predicted_values, 
                normalization=args.normalization
            )
            processed_count += 1
            
            # Collect errors for overall metrics
            all_abs_errors.extend(abs_errors)
            components = ['calories', 'mass', 'fat', 'carbs', 'protein']
            for i, comp in enumerate(components):
                error = abs(predicted_values[comp] - actual_values[comp])
                component_errors[comp].append(error)
                component_norm_errors[comp].append(norm_errors[comp])
                
                # Store normalization factors for each component
                if comp not in norm_factors:
                    norm_factors[comp] = dish_norm_factors[comp]
            
            # Add a line break between dishes
            if dish_id != dish_ids[-1]:
                print("\n")
                
        except Exception as e:
            print(f"Error processing {dish_id}: {e}")
            failed_count += 1
    
    # Calculate overall metrics across all dishes
    if all_abs_errors:
        overall_mae = np.mean(all_abs_errors)
        overall_rmse = np.sqrt(np.mean(np.square(all_abs_errors)))
        
        # Calculate normalized overall metrics
        all_norm_errors_flat = []
        for comp in component_norm_errors:
            all_norm_errors_flat.extend(component_norm_errors[comp])
        
        overall_nmae = np.mean(all_norm_errors_flat)
        overall_nrmse = np.sqrt(np.mean(np.square(all_norm_errors_flat)))
        
        # Calculate component-specific metrics
        component_maes = {}
        component_rmses = {}
        component_nmaes = {}
        component_nrmses = {}
        
        for comp in component_errors:
            if component_errors[comp]:
                component_maes[comp] = np.mean(component_errors[comp])
                component_rmses[comp] = np.sqrt(np.mean(np.square(component_errors[comp])))
                component_nmaes[comp] = np.mean(component_norm_errors[comp])
                component_nrmses[comp] = np.sqrt(np.mean(np.square(component_norm_errors[comp])))
        
        print("\n" + "=" * 80)
        print(f"OVERALL METRICS ACROSS ALL DISHES (Normalization: {args.normalization})")
        print("=" * 80)
        print(f"Processed dishes: {processed_count}/{len(dish_ids)} (Failed: {failed_count})")
        print(f"Label scaling: {args.label_scaling}")
        print(f"Overall MAE: {overall_mae:.2f} | Overall RMSE: {overall_rmse:.2f}")
        print(f"Overall NMAE: {overall_nmae:.4f} | Overall NRMSE: {overall_nrmse:.4f}")
        
        # Print component-specific results
        print("-" * 80)
        print(f"{'Component':<12} | {'MAE':>8} | {'RMSE':>8} | {'NMAE':>8} | {'NRMSE':>8} | {'Norm Factor':>12}")
        print("-" * 80)
        
        for comp in ['calories', 'mass', 'fat', 'carbs', 'protein']:
            print(f"{comp.capitalize():<12} | {component_maes[comp]:>8.2f} | {component_rmses[comp]:>8.2f} | {component_nmaes[comp]:>8.4f} | {component_nrmses[comp]:>8.4f} | {norm_factors[comp]:>12.2f}")
            
        print("=" * 80)

if __name__ == "__main__":
    main() 
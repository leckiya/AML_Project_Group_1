import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from models.efficientnet_model import get_model, get_v2_model


def predict_nutrients(model, image_path):
    """
    Predict nutrients for a given image
    
    Args:
        model (nn.Module): Trained model
        image_path (str): Path to the image file
        
    Returns:
        dict: Predicted nutrient values
    """
    # Only CUDA is supported
    device = torch.device("cuda")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)
    
    predictions = predictions.cpu().numpy()[0]
    
    return {
        'mass': predictions[0],
        'calories': predictions[1],
        'fat': predictions[2],
        'carbs': predictions[3],
        'protein': predictions[4]
    }


def visualize_prediction(image_path, predictions):
    """
    Create a visualization of the prediction
    
    Args:
        image_path (str): Path to the image file
        predictions (dict): Dictionary of predicted nutrients
        
    Returns:
        matplotlib.figure.Figure: Figure with the visualization
    """
    image = Image.open(image_path).convert('RGB')
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(image)
    ax.axis('off')
    
    info_text = "\n".join([
        f"Mass: {predictions['mass']:.1f} g",
        f"Calories: {predictions['calories']:.1f} kcal",
        f"Fat: {predictions['fat']:.1f} g",
        f"Carbs: {predictions['carbs']:.1f} g",
        f"Protein: {predictions['protein']:.1f} g"
    ])
    
    plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Food Nutrient Prediction')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the food image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--efficientnet_version', type=str, default='b0', help='EfficientNet version')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the visualization')
    args = parser.parse_args()
    
    # Set device to CUDA only
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for inference.")
    device = torch.device("cuda")
    print("Using device: cuda")
    
    # Load model
    if args.efficientnet_version.startswith("v2_"):
        model = get_v2_model(
            efficientnet_version=args.efficientnet_version[3:],
            num_outputs=5,
            pretrained=False
        )
    else:
        model = get_model(
            efficientnet_version=args.efficientnet_version,
            num_outputs=5,
            pretrained=False
        )
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Model loaded from {args.model_path}")
    
    # Make prediction
    predictions = predict_nutrients(model, args.image_path)
    
    # Print results
    print("\n===== Nutrient Predictions =====")
    for nutrient, value in predictions.items():
        print(f"{nutrient.capitalize()}: {value:.2f}")
    
    # Create visualization
    fig = visualize_prediction(args.image_path, predictions)
    
    # Save or display
    if args.output_path:
        fig.savefig(args.output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {args.output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main() 
# Food Image Calorie and Nutrient Predictor

This project implements a deep learning model based on EfficientNet to predict calories and nutrients (fat, carbs, protein) from food images using the Nutrition5k dataset.

## Project Structure

```
.
├── src/
│   ├── data/
│   │   ├── dataset.py         # Dataset loader
│   │   └── preprocess.py      # Data preprocessing
│   ├── models/
│   │   └── efficientnet_model.py  # EfficientNet model definition
│   ├── utils/
│   │   ├── training.py        # Training utilities
│   │   └── scaler.py          # Label scaling utilities
│   ├── evaluation/
│   │   └── metrics.py         # Evaluation metrics
│   ├── train.py               # Main training script
│   ├── evaluate_model.py      # Model evaluation script
│   ├── dish_inference.py      # Dish-specific inference 
│   └── inference.py           # General inference script
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## Dataset

This project uses the [Nutrition5k dataset](https://github.com/google-research-datasets/Nutrition5k), which includes:
- RGB images of food dishes
- Depth images
- Nutritional information (mass, calories, fat, carbs, protein)

## Setup

1. Clone the repository and navigate to the project directory.
2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the Nutrition5k dataset and extract it to `nutrition5k_dataset/` directory.

## Preprocessing

To preprocess the Nutrition5k dataset:

```bash
python src/data/preprocess.py --data_dir nutrition5k_dataset --output_dir processed_data
```

Options:
- `--cafe1_only`: Process only data from cafe1 (smaller dataset)

## Training

To train the model:

```bash
python src/train.py --data_dir nutrition5k_dataset/imagery \
                    --split_dir nutrition5k_dataset/dish_ids/splits \
                    --metadata_file nutrition5k_dataset/metadata/dish_metadata_cafe1.csv \
                    --pretrained \
                    --batch_size 16 \
                    --num_epochs 30
```

Key parameters:
- `--efficientnet_version`: Version of EfficientNet to use (default: b0)
- `--pretrained`: Use pretrained weights
- `--batch_size`: Batch size for training
- `--num_epochs`: Number of epochs to train
- `--lr`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay factor (default: 1e-5)

### Label Scaling

To enable label scaling during training:

```bash
python src/train.py --label_scaling --scaler_type standard [other params]
```

Options:
- `--label_scaling`: Enable scaling of target values
- `--scaler_type`: Choose scaling method (`standard` or `minmax`)

## Evaluation

To evaluate a trained model:

```bash
python src/evaluate_model.py --model_path checkpoints/model_name/best_model.pth \
                             --data_dir nutrition5k_dataset/imagery \
                             --split_dir nutrition5k_dataset/dish_ids/splits \
                             --metadata_file nutrition5k_dataset/metadata/dish_metadata_cafe1.csv
```

Options:
- `--label_scaling`: Enable if model was trained with label scaling
- `--scaler_path`: Path to label scaler (optional, auto-detected from model directory)
- `--normalization`: Method for metric normalization (`range`, `mean`, or `std`)
- `--avg_r2`: Use averaged R² calculation

### Dish-specific Evaluation

To run inference on specific dishes:

```bash
python src/dish_inference.py --model_path checkpoints/model_name/best_model.pth \
                             --data_dir nutrition5k_dataset/imagery \
                             --metadata_file nutrition5k_dataset/metadata/dish_metadata_cafe1.csv \
                             --dish_ids dish_1565033189 dish_1556572657
```

Options:
- `--all_dishes`: Process all dishes in metadata file
- `--max_dishes`: Maximum number of dishes to process with `--all_dishes`
- `--normalization`: Method for metric normalization (`range`, `mean`, or `std`)
- `--label_scaling`: Enable if model was trained with label scaling

## General Inference

To run inference on a new food image:

```bash
python src/inference.py --image_path path/to/food_image.jpg \
                        --model_path checkpoints/best_model.pth
```

Options:
- `--output_path`: Save visualization to file
- `--efficientnet_version`: Version of EfficientNet used in trained model
- `--label_scaling`: Enable if model was trained with label scaling
- `--scaler_path`: Path to label scaler (optional)

## Evaluation Metrics

The model is evaluated with the following metrics:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R²)
- Mean Absolute Percentage Error (MAPE)
- Normalized RMSE (NRMSE)
- Normalized MAE (NMAE)

### Normalization Methods

The NRMSE and NMAE metrics can be calculated using different normalization approaches:
- `range`: Normalize by max-min range of the target
- `mean`: Normalize by mean value of the target
- `std`: Normalize by standard deviation of the target

## Results

After training, results are saved to the `results/` directory:
- Prediction vs. target plots
- Evaluation metrics for each nutrient (mass, calories, fat, carbs, protein)
- Overall metrics
- Normalized metrics for comparability between different nutrients

## TensorBoard

View training progress with TensorBoard:

```bash
tensorboard --logdir logs
```

## References

- [Nutrition5k Dataset](https://github.com/google-research-datasets/Nutrition5k)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) 
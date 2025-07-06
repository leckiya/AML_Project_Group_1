import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import R2Score, MeanIoU
import itertools
from tensorflow.keras.metrics import TopKCategoricalAccuracy
import time
import sys
import psutil
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tabulate import tabulate

OUTPUT_DIR = 'results_v6'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Global Constants ---
DATASET_DIR = 'nutrition5k_dataset'
METADATA_DIR = os.path.join(DATASET_DIR, 'metadata')
IMAGERY_DIR = os.path.join(DATASET_DIR, 'imagery', 'realsense_overhead')
SPLIT_DIR = os.path.join(DATASET_DIR, 'dish_ids', 'splits')

IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32

# --- Custom Loss Function ---
def normalized_metrics_loss(y_true, y_pred):
    """
    Custom loss function that combines normalized MAE, RMSE, and R2 across 5 nutritional categories.
    The loss is designed to be minimized, so we use (1 - R2) for R2 component.
    """
    # Calculate metrics for each nutrient (5 categories)
    mae_losses = []
    rmse_losses = []
    r2_losses = []
    
    for i in range(5):  # 5 nutrients: calories, mass, fat, carb, protein
        y_true_nutrient = y_true[:, i]
        y_pred_nutrient = y_pred[:, i]
        
        # MAE (normalized by mean of true values)
        mae = tf.reduce_mean(tf.abs(y_true_nutrient - y_pred_nutrient))
        mae_normalized = mae / (tf.reduce_mean(tf.abs(y_true_nutrient)) + 1e-8)
        mae_losses.append(mae_normalized)
        
        # RMSE (normalized by mean of true values)
        mse = tf.reduce_mean(tf.square(y_true_nutrient - y_pred_nutrient))
        rmse = tf.sqrt(mse + 1e-8)
        rmse_normalized = rmse / (tf.reduce_mean(tf.abs(y_true_nutrient)) + 1e-8)
        rmse_losses.append(rmse_normalized)
        
        # R2 component (1 - R2, since we want to minimize loss)
        # R2 = 1 - (SS_res / SS_tot)
        ss_res = tf.reduce_sum(tf.square(y_true_nutrient - y_pred_nutrient))
        ss_tot = tf.reduce_sum(tf.square(y_true_nutrient - tf.reduce_mean(y_true_nutrient)))
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        r2_loss = 1 - r2  # Convert to loss (minimize 1-R2)
        r2_losses.append(r2_loss)
    
    # Average across all nutrients
    avg_mae = tf.reduce_mean(mae_losses)
    avg_rmse = tf.reduce_mean(rmse_losses)
    avg_r2_loss = tf.reduce_mean(r2_losses)
    
    # Combine all metrics (equal weights)
    total_loss = avg_mae + avg_rmse + avg_r2_loss
    
    return total_loss

# --- 1. Data Loading and Preprocessing ---
# Only RGB images are used: imagery/realsense_overhead/<dish_id>/rgb.png
# Both cafe1 and cafe2 metadata are always loaded for the full dataset

def load_full_data():
    """
    Loads the entire Nutrition5k dataset, using both cafe1 and cafe2 metadata, and only RGB images.
    Returns a DataFrame with all available samples and their metadata.
    """
    print(f"--- Loading FULL Nutrition5k dataset (ignoring split files) ---")
    # 1. Load all metadata
    metadata_files = [os.path.join(METADATA_DIR, f) for f in ['dish_metadata_cafe1.csv', 'dish_metadata_cafe2.csv']]
    df_list = []
    col_names = ['dish_id', 'total_calories', 'total_mass', 'total_fat', 'total_carb', 'total_protein', 'num_ingrs']
    for f_path in metadata_files:
        try:
            df = pd.read_csv(f_path, header=None, usecols=range(7), names=col_names, on_bad_lines='skip', engine='python')
            df_list.append(df)
        except FileNotFoundError:
            print(f"Warning: Metadata file not found at {f_path}")
            continue
    if not df_list:
        print("Error: No metadata files found. Aborting.")
        return pd.DataFrame()
    metadata = pd.concat(df_list, ignore_index=True)
    # 2. Construct file paths and verify existence
    data = []
    for _, row in metadata.iterrows():
        dish_id = row['dish_id']
        img_path = os.path.join(IMAGERY_DIR, dish_id, 'rgb.png')
        if os.path.exists(img_path):
            data.append({
                'file_path': img_path,
                'dish_id': dish_id,
                'calories': row['total_calories'],
                'mass': row['total_mass'],
                'fat': row['total_fat'],
                'carb': row['total_carb'],
                'protein': row['total_protein']
            })
    df = pd.DataFrame(data)
    print(f"Found {len(df)} images in the full dataset.")
    if df.empty:
        print(f"Warning: No data loaded. Check paths and file existence.")
    return df

def explore_and_process_data(df, split_name, caps=None):
    """
    Performs data exploration and preprocessing.
    For the test set, it uses the caps calculated from the training set.
    """
    print(f"\n--- Exploring and Processing {split_name} data ---")
    
    nutritional_cols = ['calories', 'mass', 'fat', 'carb', 'protein']
    
    # 1. Descriptive Statistics
    print("Descriptive Statistics for Nutritional Information (before processing):")
    print(df[nutritional_cols].describe())
    
    # 2. Check for missing values
    print("\nMissing Values:")
    print(df[nutritional_cols].isnull().sum())
    
    # 3. Visualize distributions
    print("\nGenerating distribution plots...")
    for col in nutritional_cols:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True, bins=50)
        plt.title(f'Distribution of {col} in {split_name} data (Original)')
        plt.savefig(os.path.join(OUTPUT_DIR, f'{split_name}_{col}_distribution_original.png'))
        plt.show()
        plt.close()
    
    # 4. Outlier Capping
    if split_name == 'train':
        print("\nCalculating caps and capping outliers at the 99th percentile for training data...")
        caps = {}
        for col in nutritional_cols:
            percentile_99 = df[col].quantile(0.99)
            caps[col] = percentile_99
            df[col] = df[col].apply(lambda x: min(x, percentile_99))
            print(f"Capped '{col}' at: {percentile_99:.2f}")

        print("\nDescriptive Statistics after Outlier Capping:")
        print(df[nutritional_cols].describe())
        return df, caps
    else: # For 'test' or 'validation' data
        if caps:
            print("\nCapping outliers in test data based on training set caps...")
            for col in nutritional_cols:
                df[col] = df[col].apply(lambda x: min(x, caps[col]))
            print("\nDescriptive Statistics after Outlier Capping:")
            print(df[nutritional_cols].describe())
        else:
            print("\nWarning: No caps provided for test data processing. Outliers not handled.")
        return df, None

# --- 2. Model Building ---
# ResNet50 is used as a feature extractor. It is a deep convolutional neural network with residual connections, allowing for very deep architectures without vanishing gradients. The model computes features by applying a series of convolutional, batch normalization, and ReLU layers, with skip connections that add the input of a block to its output. The final feature vector is obtained by global average pooling over the last convolutional feature maps.

def build_regression_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), pooling='avg')
    base_model.trainable = False # We will only use it for feature extraction
    return base_model

def build_regression_head(input_shape):
    """Builds the small regression model that trains on extracted features."""
    model = tf.keras.Sequential([
        Input(shape=input_shape),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dense(5, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=normalized_metrics_loss, metrics=[R2Score()])
    return model

def build_baseline_regression_head(input_shape):
    # Baseline: minimal head, no regularization, high learning rate, few epochs
    model = tf.keras.Sequential([
        Input(shape=input_shape),
        Dense(5, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=1e-2), loss=normalized_metrics_loss, metrics=[R2Score()])
    return model

# --- 3. Task Execution ---

def extract_features(df, model):
    """
    Uses the base ResNet50 model to extract features from all images.
    """
    print("\n--- Extracting Features ---")
    
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    
    generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col='file_path',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode=None, # We only want the images
        shuffle=False # Keep order for matching with labels
    )
    
    features = model.predict(generator, steps=len(generator), verbose=1)
    print(f"Extracted features with shape: {features.shape}")
    return features

def run_regression_task(train_features, train_labels, test_features, test_labels, scaler, model_type='improved', epochs=200):
    # model_type: 'baseline' or 'improved'
    # epochs: number of epochs to train
    X_train, X_val, y_train, y_val = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=42
    )
    if model_type == 'baseline':
        model = build_baseline_regression_head(input_shape=(train_features.shape[1],))
    else:
        model = build_regression_head(input_shape=(train_features.shape[1],))
    early_stopping = EarlyStopping(monitor='val_r2_score', patience=40, restore_best_weights=True, verbose=1, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_r2_score', factor=0.2, patience=20, min_lr=1e-7, verbose=1, mode='max')
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=[early_stopping, reduce_lr] if model_type=='improved' else []
    )
    predictions_scaled = model.predict(test_features)
    y_true_scaled = test_labels
    predictions = scaler.inverse_transform(predictions_scaled)
    y_true = scaler.inverse_transform(y_true_scaled)
    # Per-nutrient metrics
    metrics = {}
    nutrients = ['Calories', 'Mass', 'Fat', 'Carb', 'Protein']
    for i, nutrient in enumerate(nutrients):
        mse = mean_squared_error(y_true[:, i], predictions[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true[:, i], predictions[:, i])
        r2 = r2_score(y_true[:, i], predictions[:, i])
        mape = np.mean(np.abs((y_true[:, i] - predictions[:, i]) / (y_true[:, i] + 1e-8))) * 100
        value_range = y_true[:, i].max() - y_true[:, i].min()
        nrmse = rmse / (value_range + 1e-8)
        nmae = mae / (value_range + 1e-8)
        norm_factor = np.mean(np.abs(y_true[:, i]))
        metrics[nutrient] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'NRMSE': nrmse,
            'NMAE': nmae,
            'Norm factor': norm_factor
        }
    # Overall metrics (mean across all nutrients)
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    mape = np.mean(np.abs((y_true - predictions) / (y_true + 1e-8))) * 100
    value_range = y_true.max() - y_true.min()
    nrmse = rmse / (value_range + 1e-8)
    nmae = mae / (value_range + 1e-8)
    norm_factor = np.mean(np.abs(y_true))
    metrics['Overall'] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'NRMSE': nrmse,
        'NMAE': nmae,
        'Norm factor': norm_factor
    }
    return model, metrics, predictions, y_true, history

def plot_training_history(history, model_type, output_dir):
    """
    Plot training history for the custom loss function.
    """
    plt.figure(figsize=(15, 5))
    
    # Plot loss
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_type} Model - Custom Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot R2 Score
    plt.subplot(1, 3, 2)
    plt.plot(history.history['r2_score'], label='Training R2')
    plt.plot(history.history['val_r2_score'], label='Validation R2')
    plt.title(f'{model_type} Model - R2 Score')
    plt.xlabel('Epoch')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.grid(True)
    
    # Plot learning rate (if available)
    plt.subplot(1, 3, 3)
    if 'lr' in history.history:
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.title(f'{model_type} Model - Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
    else:
        plt.text(0.5, 0.5, 'Learning Rate\nNot Available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title(f'{model_type} Model - Learning Rate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_type.lower()}_training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def generate_summary_report(regression_history):
    """
    Generates a summary report of all tasks.
    """
    print("\n--- Generating Summary Report ---")
    
    report_path = os.path.join(OUTPUT_DIR, 'summary_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("======================================\n")
        print("======================================")
        f.write("      NUTRITION ANALYSIS REPORT      \n")
        print("      NUTRITION ANALYSIS REPORT      ")
        f.write("======================================\n\n")
        print("======================================\n")
        # --- Regression Summary ---
        f.write("--- Calorie/Nutrient Estimation Task ---\n")
        print("--- Calorie/Nutrient Estimation Task ---")
        f.write("--- Using Custom Normalized Metrics Loss ---\n")
        print("--- Using Custom Normalized Metrics Loss ---")
        if regression_history and regression_history.history:
            best_epoch = np.argmin(regression_history.history['val_loss']) + 1
            f.write(f"Best model saved from epoch: {best_epoch}\n")
            print(f"Best model saved from epoch: {best_epoch}")
            f.write(f"Final custom loss: {regression_history.history['loss'][-1]:.6f}\n")
            print(f"Final custom loss: {regression_history.history['loss'][-1]:.6f}")
            f.write(f"Final validation custom loss: {regression_history.history['val_loss'][-1]:.6f}\n")
            print(f"Final validation custom loss: {regression_history.history['val_loss'][-1]:.6f}")
            reg_report_path = os.path.join(OUTPUT_DIR, 'regression_report.csv')
            if os.path.exists(reg_report_path):
                report_df = pd.read_csv(reg_report_path)
                overall_metrics = report_df[report_df['Target'] == 'Overall']
                calories_metrics = report_df[report_df['Target'] == 'Calories']
                if not overall_metrics.empty and not calories_metrics.empty:
                    r2_val = overall_metrics['R-squared (R2)'].values[0]
                    r2 = float(r2_val) if r2_val != '-' else -1.0
                    mae = calories_metrics['Mean Absolute Error (MAE)'].values[0]
                    rmse = calories_metrics['Root Mean Squared Error (RMSE)'].values[0]
                    f.write(f"  - R-squared (Overall): {r2:.4f}\n")
                    print(f"  - R-squared (Overall): {r2:.4f}")
                    f.write(f"  - Calories MAE: {mae:.4f}\n")
                    print(f"  - Calories MAE: {mae:.4f}")
                    f.write(f"  - Calories RMSE: {rmse:.4f}\n")
                    print(f"  - Calories RMSE: {rmse:.4f}")
            else:
                f.write("  - Regression report file not found.\n")
                print("  - Regression report file not found.")
        else:
            f.write("  - Regression task was skipped or failed.\n")
            print("  - Regression task was skipped or failed.")
        f.write("\n")
        print()

    print(f"Summary report saved to {report_path}")

def generate_final_report(train_df, test_df, regression_metrics):
    report_path = os.path.join(OUTPUT_DIR, 'final_report.txt')
    with open(report_path, 'w') as f:
        # 1. Executive Summary
        f.write("1. Executive Summary\n")
        print("1. Executive Summary")
        f.write("====================\n")
        print("====================")
        f.write("This project aims to automate food calorie estimation from images using a deep learning pipeline for calorie regression. The approach achieves accurate calorie prediction, with all results, metrics, and visualizations saved for reproducibility.\n\n")
        print("This project aims to automate food calorie estimation from images using a deep learning pipeline for calorie regression. The approach achieves accurate calorie prediction, with all results, metrics, and visualizations saved for reproducibility.\n")
        # 2. Introduction
        f.write("2. Introduction\n")
        print("2. Introduction")
        f.write("===============\n")
        print("===============" )
        f.write("Motivation: Health awareness and the challenge of accurate calorie tracking.\n")
        print("Motivation: Health awareness and the challenge of accurate calorie tracking.")
        f.write("Problem: Automate calorie estimation from food images.\n")
        print("Problem: Automate calorie estimation from food images.")
        f.write("Objectives: Build a robust, extensible pipeline for food calorie estimation.\n")
        print("Objectives: Build a robust, extensible pipeline for food calorie estimation.")
        f.write("Business Applicability: Enables dietary feedback, mobile health, and future extensions.\n\n")
        print("Business Applicability: Enables dietary feedback, mobile health, and future extensions.\n")
        # 3. Related Work / Literature Review
        f.write("3. Related Work / Literature Review\n")
        f.write("===================================\n")
        f.write("- Existing tools: MyFitnessPal, CalorieMama.\n")
        f.write("- Academic/industry research: Deep learning for food calorie estimation.\n")
        f.write("- Gaps: Manual entry, limited generalization, lack of end-to-end automation.\n")
        f.write("- Our originality: Robust metrics, extensible design.\n\n")
        print("- Existing tools: MyFitnessPal, CalorieMama.")
        print("- Academic/industry research: Deep learning for food calorie estimation.")
        print("- Gaps: Manual entry, limited generalization, lack of end-to-end automation.")
        print("- Our originality: Robust metrics, extensible design.")
        f.write("\n")
        # 4. Dataset Description
        f.write("4. Dataset Description\n")
        f.write("======================\n")
        f.write(f"Origin: Nutrition5k Dataset (see README).\n")
        f.write(f"Train images: {len(train_df)}, Test images: {len(test_df)}\n")
        f.write(f"Categories: {train_df['dish_id'].nunique()}\n")
        f.write("Structure: imagery/metadata/splits.\n")
        f.write("Preprocessing: resizing to 224x224, normalization, outlier capping, augmentation.\n")
        f.write("Biases: Some classes have few samples; test set may contain unseen classes.\n\n")
        print(f"Origin: Nutrition5k Dataset (see README).")
        print(f"Train images: {len(train_df)}, Test images: {len(test_df)}")
        print(f"Categories: {train_df['dish_id'].nunique()}")
        print("Structure: imagery/metadata/splits.")
        print("Preprocessing: resizing to 224x224, normalization, outlier capping, augmentation.")
        print("Biases: Some classes have few samples; test set may contain unseen classes.")
        f.write("\n")
        # 5. Methodology
        f.write("5. Methodology\n")
        f.write("==============\n")
        f.write("5.1 Task Definition: Calorie regression.\n")
        f.write("5.2 Model Selection: ResNet50 for feature extraction, small regression head.\n")
        f.write("Training: Custom normalized metrics loss (MAE + RMSE + R2), Adam optimizer, regularization via dropout and batch norm.\n\n")
        print("5.1 Task Definition: Calorie regression.")
        print("5.2 Model Selection: ResNet50 for feature extraction, small regression head.")
        print("Training: Custom normalized metrics loss (MAE + RMSE + R2), Adam optimizer, regularization via dropout and batch norm.")
        f.write("\n")
        # 6. Implementation Details
        f.write("6. Implementation Details\n")
        f.write("========================\n")
        f.write("Frameworks: TensorFlow, Keras, scikit-learn, matplotlib, seaborn.\n")
        f.write("Parameters: batch size 32, learning rate 1e-3, epochs up to 200, early stopping.\n")
        f.write("Loss Function: Custom normalized metrics loss combining MAE, RMSE, and R2 across 5 nutrients.\n")
        f.write("Hardware: [Specify GPU/CPU/Colab/local].\n")
        f.write("Challenges: model overfitting, outlier handling, custom loss optimization.\n\n")
        print("Frameworks: TensorFlow, Keras, scikit-learn, matplotlib, seaborn.")
        print("Parameters: batch size 32, learning rate 1e-3, epochs up to 200, early stopping.")
        print("Loss Function: Custom normalized metrics loss combining MAE, RMSE, and R2 across 5 nutrients.")
        print("Hardware: [Specify GPU/CPU/Colab/local].")
        print("Challenges: model overfitting, outlier handling, custom loss optimization.")
        f.write("\n")
        # 7. Results & Evaluation
        f.write("7. Results & Evaluation\n")
        f.write("=======================\n")
        f.write("7.1 Calorie Estimation Metrics\n")
        f.write(f"  - MAE: {regression_metrics['mae']:.2f}\n")
        print(f"  - MAE: {regression_metrics['mae']:.2f}")
        f.write(f"  - MSE: {regression_metrics['mse']:.2f}\n")
        print(f"  - MSE: {regression_metrics['mse']:.2f}")
        f.write(f"  - RMSE: {regression_metrics['rmse']:.2f}\n")
        print(f"  - RMSE: {regression_metrics['rmse']:.2f}")
        f.write(f"  - R²: {regression_metrics['r2']:.4f}\n")
        print(f"  - R²: {regression_metrics['r2']:.4f}")
        f.write("7.2 Comparison and Insights\n")
        f.write("  - [Add comparison table and discussion here] \n\n")
        # 8. Innovation & Future Work
        f.write("8. Innovation & Future Work\n")
        f.write("===========================\n")
        f.write("- Extensions: personalized feedback, portion size, mobile deployment.\n")
        f.write("- Innovations: multi-task learning, ensembles, multimodal input.\n")
        f.write("- Future: expand dataset, real-world deployment, user feedback.\n\n")
        print("- Extensions: personalized feedback, portion size, mobile deployment.")
        print("- Innovations: multi-task learning, ensembles, multimodal input.")
        print("- Future: expand dataset, real-world deployment, user feedback.")
        f.write("\n")
        # 9. Conclusion
        f.write("9. Conclusion\n")
        f.write("==============\n")
        f.write("This project demonstrates a robust, extensible pipeline for food image calorie estimation. The approach achieves strong regression results, with clear potential for real-world impact and future extension.\n\n")
        print("This project demonstrates a robust, extensible pipeline for food image calorie estimation. The approach achieves strong regression results, with clear potential for real-world impact and future extension.")
        f.write("\n")
        # 10. References
        f.write("10. References\n")
        f.write("===============\n")
        f.write("- Nutrition5k Dataset\n- ResNet: He et al., 2015\n- TensorFlow, Keras, scikit-learn\n- [Add more as needed]\n\n")
        print("- Nutrition5k Dataset")
        print("- ResNet: He et al., 2015")
        print("- TensorFlow, Keras, scikit-learn")
        print("- [Add more as needed]")
        f.write("\n")
        # 11. Appendix
        f.write("11. Appendix\n")
        f.write("==============\n")
        f.write("- Model architecture diagrams: [Add if available]\n")
        f.write("- Sample predictions: [Add screenshots if available]\n")
        f.write("- Additional metrics: See CSVs and PNGs in results/\n")
        print("- Model architecture diagrams: [Add if available]")
        print("- Sample predictions: [Add screenshots if available]")
        print("- Additional metrics: See CSVs and PNGs in results/")
        f.write("\n")
    print(f"Final report saved to {report_path}")

def save_detailed_metrics(baseline_metrics, improved_metrics, output_dir):
    """
    Save detailed metrics to CSV files for analysis.
    """
    # Save overall metrics
    overall_data = []
    for metric in ['MAE', 'RMSE', 'R2']:
        overall_data.append({
            'Metric': metric,
            'Baseline': baseline_metrics['Overall'][metric],
            'Improved': improved_metrics['Overall'][metric],
            'Improvement_Percent': ((improved_metrics['Overall'][metric] - baseline_metrics['Overall'][metric]) / 
                                  abs(baseline_metrics['Overall'][metric])) * 100
        })
    
    overall_df = pd.DataFrame(overall_data)
    overall_df.to_csv(os.path.join(output_dir, 'overall_metrics_detailed.csv'), index=False)
    
    # Save per-nutrient metrics
    nutrient_data = []
    nutrients = ['Calories', 'Mass', 'Fat', 'Carb', 'Protein']
    for nutrient in nutrients:
        for metric in ['MAE', 'RMSE', 'R2']:
            nutrient_data.append({
                'Nutrient': nutrient,
                'Metric': metric,
                'Baseline': baseline_metrics[nutrient][metric],
                'Improved': improved_metrics[nutrient][metric],
                'Improvement_Percent': ((improved_metrics[nutrient][metric] - baseline_metrics[nutrient][metric]) / 
                                      abs(baseline_metrics[nutrient][metric])) * 100
            })
    
    nutrient_df = pd.DataFrame(nutrient_data)
    nutrient_df.to_csv(os.path.join(output_dir, 'nutrient_metrics_detailed.csv'), index=False)
    
    print(f"Detailed metrics saved to {output_dir}")

def plot_predictions_vs_actual(y_true, baseline_preds, improved_preds, output_dir):
    """
    Generate scatter plots comparing predictions vs actual values for each nutrient.
    """
    nutrients = ['Calories', 'Mass', 'Fat', 'Carb', 'Protein']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, nutrient in enumerate(nutrients):
        ax = axes[i]
        
        # Plot baseline predictions
        ax.scatter(y_true[:, i], baseline_preds[:, i], alpha=0.6, label='Baseline', s=30)
        # Plot improved predictions
        ax.scatter(y_true[:, i], improved_preds[:, i], alpha=0.6, label='Improved', s=30)
        
        # Plot perfect prediction line
        min_val = min(y_true[:, i].min(), baseline_preds[:, i].min(), improved_preds[:, i].min())
        max_val = max(y_true[:, i].max(), baseline_preds[:, i].max(), improved_preds[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')
        
        ax.set_xlabel(f'Actual {nutrient}')
        ax.set_ylabel(f'Predicted {nutrient}')
        ax.set_title(f'{nutrient} Predictions vs Actual')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove the last subplot (6th position)
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions_vs_actual_all_nutrients.png'), dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def print_metrics_block(title, metrics):
    print(f"\n-- {title.upper()} --")
    print(f"MSE:    {metrics['MSE']:.4f}")
    print(f"RMSE:   {metrics['RMSE']:.4f}")
    print(f"MAE:    {metrics['MAE']:.4f}")
    print(f"R²:     {metrics['R2']:.4f}")
    print(f"MAPE:   {metrics['MAPE']:.2f}%")
    print(f"NRMSE (range): {metrics['NRMSE']:.4f}")
    print(f"NMAE (range):  {metrics['NMAE']:.4f}")
    print(f"Norm factor:   {metrics['Norm factor']:.4f}")

def save_full_metrics_csv(baseline_metrics, improved_metrics, output_dir):
    """
    Save a single CSV with all metrics (MSE, RMSE, MAE, R2, MAPE, NRMSE, NMAE, Norm factor) for both models, for each nutrient and overall.
    """
    metrics_list = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE', 'NRMSE', 'NMAE', 'Norm factor']
    nutrients = ['Overall', 'Calories', 'Mass', 'Fat', 'Carb', 'Protein']
    rows = []
    for nutrient in nutrients:
        row = {'Nutrient': nutrient}
        for metric in metrics_list:
            row[f'Baseline {metric}'] = baseline_metrics[nutrient][metric]
            row[f'Improved {metric}'] = improved_metrics[nutrient][metric]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'full_metrics_comparison.csv'), index=False)
    print(f"Full metrics comparison saved to {os.path.join(output_dir, 'full_metrics_comparison.csv')}")

def main():
    # Load full data (ignore split files)
    full_df = load_full_data()
    if full_df.empty:
        print("Could not load data. Exiting.")
        return
    # Filter to classes with at least 2 samples
    class_counts = full_df['dish_id'].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    filtered_df = full_df[full_df['dish_id'].isin(valid_classes)].copy()
    if len(filtered_df) == 0:
        print("No classes with at least 2 samples. Using all available classes (no filtering).")
        filtered_df = full_df.copy()
    else:
        print(f"After filtering: {len(filtered_df)} samples, {filtered_df['dish_id'].nunique()} classes with >=2 samples.")
    benchmark_dishes = ['dish_1565033189', 'dish_1556572657']
    benchmark_rows = filtered_df[filtered_df['dish_id'].isin(benchmark_dishes)]
    non_benchmark_df = filtered_df[~filtered_df['dish_id'].isin(benchmark_dishes)]
    train_df, test_df = train_test_split(non_benchmark_df, test_size=0.2, random_state=42)
    test_df = pd.concat([test_df, benchmark_rows]).drop_duplicates().reset_index(drop=True)
    nutritional_cols = ['calories', 'mass', 'fat', 'carb', 'protein']
    # Preprocess
    train_df, caps = explore_and_process_data(train_df, 'train')
    test_df, _ = explore_and_process_data(test_df, 'test', caps=caps)
    # Feature extraction
    base_model = build_regression_model()
    train_features = extract_features(train_df, base_model)
    test_features = extract_features(test_df, base_model)
    # Scale labels
    scaler = StandardScaler()
    train_labels_scaled = scaler.fit_transform(train_df[nutritional_cols])
    test_labels_scaled = scaler.transform(test_df[nutritional_cols])
    # Baseline model
    baseline_model, baseline_metrics, baseline_preds, baseline_y_true, baseline_history = run_regression_task(
        train_features, train_labels_scaled, test_features, test_labels_scaled, scaler, model_type='baseline', epochs=10)
    # Improved model
    improved_model, improved_metrics, improved_preds, improved_y_true, improved_history = run_regression_task(
        train_features, train_labels_scaled, test_features, test_labels_scaled, scaler, model_type='improved', epochs=200)
    
    # Plot training histories
    plot_training_history(baseline_history, 'Baseline', OUTPUT_DIR)
    plot_training_history(improved_history, 'Improved', OUTPUT_DIR)
    
    # Save models
    baseline_model.save(os.path.join(OUTPUT_DIR, 'baseline_model.keras'))
    improved_model.save(os.path.join(OUTPUT_DIR, 'improved_model.keras'))
    
    # Save detailed metrics
    save_detailed_metrics(baseline_metrics, improved_metrics, OUTPUT_DIR)
    
    # Generate predictions vs actual plots
    plot_predictions_vs_actual(baseline_y_true, baseline_preds, improved_preds, OUTPUT_DIR)
    
    # Table 1: Overall metrics and improvement
    print("\nTable 1: Overall Model Performance")
    table1 = [
        ["MAE", baseline_metrics['Overall']['MAE'], improved_metrics['Overall']['MAE'],
         f"{100*(improved_metrics['Overall']['MAE']-baseline_metrics['Overall']['MAE'])/abs(baseline_metrics['Overall']['MAE']):.1f}%"],
        ["RMSE", baseline_metrics['Overall']['RMSE'], improved_metrics['Overall']['RMSE'],
         f"{100*(improved_metrics['Overall']['RMSE']-baseline_metrics['Overall']['RMSE'])/abs(baseline_metrics['Overall']['RMSE']):.1f}%"],
        ["R2", baseline_metrics['Overall']['R2'], improved_metrics['Overall']['R2'],
         f"{100*(improved_metrics['Overall']['R2']-baseline_metrics['Overall']['R2'])/abs(baseline_metrics['Overall']['R2']):.1f}%"]
    ]
    print(tabulate(table1, headers=["Metric", "Baseline Model", "Fine-tuned Model", "Improvement"], floatfmt=".4f"))
    pd.DataFrame(table1, columns=["Metric", "Baseline Model", "Fine-tuned Model", "Improvement"]).to_csv(os.path.join(OUTPUT_DIR, 'table1_overall_metrics.csv'), index=False)
    # Table 2: Per-nutrient metrics
    print("\nTable 2: Per-Nutrient Performance (Test Set)")
    table2 = []
    for nutrient in ['Calories', 'Mass', 'Fat', 'Carb', 'Protein']:
        table2.append([
            nutrient,
            baseline_metrics[nutrient]['MAE'], improved_metrics[nutrient]['MAE'],
            baseline_metrics[nutrient]['RMSE'], improved_metrics[nutrient]['RMSE'],
            baseline_metrics[nutrient]['R2'], improved_metrics[nutrient]['R2']
        ])
    print(tabulate(table2, headers=["Nutrient", "Baseline MAE", "Improved MAE", "Baseline RMSE", "Improved RMSE", "Baseline R2", "Improved R2"], floatfmt=".4f"))
    pd.DataFrame(table2, columns=["Nutrient", "Baseline MAE", "Improved MAE", "Baseline RMSE", "Improved RMSE", "Baseline R2", "Improved R2"]).to_csv(os.path.join(OUTPUT_DIR, 'table2_per_nutrient_metrics.csv'), index=False)
    # Table 3: Benchmark dish predictions (improved readable format)
    print("\nTable 3: Benchmark Dish Predictions (Actual vs Predicted)")
    for dish_id in benchmark_dishes:
        dish_row = test_df[test_df['dish_id'] == dish_id]
        if dish_row.empty:
            print(f"Benchmark dish {dish_id} not found in test set.")
            continue
        dish_features = extract_features(dish_row, base_model)
        dish_labels = dish_row[nutritional_cols].values
        pred_baseline = scaler.inverse_transform(baseline_model.predict(dish_features))
        pred_improved = scaler.inverse_transform(improved_model.predict(dish_features))
        print(f"\nDish: {dish_id}")
        print(f"{'Nutrient':<10} {'Actual':>10} {'Baseline Pred':>18} {'Improved Pred':>18}")
        for i, nutrient in enumerate(nutritional_cols):
            print(f"{nutrient.capitalize():<10} {dish_labels[0, i]:>10.2f} {pred_baseline[0, i]:>18.2f} {pred_improved[0, i]:>18.2f}")
    # Print metrics in screenshot format for improved model
    print("\n========================\nDETAILED METRICS (IMPROVED MODEL)\n========================")
    print_metrics_block('OVERALL', improved_metrics['Overall'])
    for nutrient in ['Calories', 'Mass', 'Fat', 'Carb', 'Protein']:
        print_metrics_block(nutrient, improved_metrics[nutrient])
    # Print metrics in screenshot format for baseline model
    print("\n========================\nDETAILED METRICS (BASELINE MODEL)\n========================")
    print_metrics_block('OVERALL', baseline_metrics['Overall'])
    for nutrient in ['Calories', 'Mass', 'Fat', 'Carb', 'Protein']:
        print_metrics_block(nutrient, baseline_metrics[nutrient])
    # Save full metrics CSV with all metrics for both models
    save_full_metrics_csv(baseline_metrics, improved_metrics, OUTPUT_DIR)
    print("\n--- Pipeline Finished ---")
    print(f"All models, reports, and visualizations saved in '{OUTPUT_DIR}' directory.")


if __name__ == '__main__':
    main()
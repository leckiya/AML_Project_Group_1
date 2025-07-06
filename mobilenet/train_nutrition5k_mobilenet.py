import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import random

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

print(pd.__version__)

# 1. Load and merge metadata
cafe1 = pd.read_csv('metadata/dish_metadata_cafe1.csv', header=None, on_bad_lines='skip')
cafe2 = pd.read_csv('metadata/dish_metadata_cafe2.csv', header=None, on_bad_lines='skip')
col_names = ['dish_id', 'calories', 'weight', 'protein', 'fat', 'carbs']
cafe1 = cafe1.iloc[:, :6]
cafe2 = cafe2.iloc[:, :6]
cafe1.columns = col_names
cafe2.columns = col_names
meta = pd.concat([cafe1, cafe2], ignore_index=True)
meta['img_path'] = meta['dish_id'].apply(lambda x: f'imagery/realsense_overhead/{x}/rgb.png')
meta = meta[meta['img_path'].apply(os.path.exists)]
print(f'Total available samples: {len(meta)}')

# 2. Load images and labels
IMG_SIZE = 224
def load_img(path):
    img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    return np.array(img)

# Create label in the order of calories, mass, fat, carbs, protein
X = np.stack([load_img(p) for p in meta['img_path']])
y = meta[['calories', 'weight', 'fat', 'carbs', 'protein']].values.astype(np.float32)

# 3. Label normalization
scaler = StandardScaler()
y_scaled = scaler.fit_transform(y)

# 4. Train/validation split (fixed random seed)
X_train, X_val, y_train, y_val = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# 5. Dataset creation
BATCH_SIZE = 64
def preprocess(x, y):
    x = tf.cast(x, tf.float32)
    x = preprocess_input(x)
    return x, y

dstrain = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).map(preprocess).batch(BATCH_SIZE).prefetch(1)
dsval = tf.data.Dataset.from_tensor_slices((X_val, y_val)).map(preprocess).batch(BATCH_SIZE).prefetch(1)

# 6. Model definition (with increased Dropout)
def build_model():
    base = MobileNetV2(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(5)(x)  # Predict 5 values: calories, mass, fat, carbs, protein
    model = Model(base.input, outputs)
    return model, base

# 7. Two-step training: base freeze, then fine-tuning
model, base = build_model()

# Step 1: Freeze base, train top layers
base.trainable = False  # base freeze
model.compile(optimizer=Adam(1e-3), loss='mse')
early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(dstrain, validation_data=dsval, epochs=30, callbacks=[early], verbose=2)

# Step 2: Unfreeze only the last 30 layers of the base for fine-tuning
for layer in base.layers[:-30]:
    layer.trainable = False
for layer in base.layers[-30:]:
    layer.trainable = True
model.compile(optimizer=Adam(1e-4), loss='mse')
model.fit(dstrain, validation_data=dsval, epochs=20, callbacks=[early], verbose=2)

# 8. Evaluation and result restoration
y_val_pred = model.predict(dsval)

# Inverse transform to original units
y_val_pred_inv = scaler.inverse_transform(y_val_pred)
y_val_true_inv = scaler.inverse_transform(y_val)

# MAE, RMSE, R2 (original units)
mae = mean_absolute_error(y_val_true_inv, y_val_pred_inv, multioutput='raw_values')
rmse = np.sqrt(mean_squared_error(y_val_true_inv, y_val_pred_inv, multioutput='raw_values'))
r2 = r2_score(y_val_true_inv, y_val_pred_inv, multioutput='raw_values')

# NRMSE, NMAE calculation (normalized by range)
y_range = np.max(y_val_true_inv, axis=0) - np.min(y_val_true_inv, axis=0)
nrmse = rmse / y_range
nmae = mae / y_range

# Overall metrics
mae_overall = np.mean(mae)
rmse_overall = np.mean(rmse)
nrmse_overall = np.mean(nrmse)
nmae_overall = np.mean(nmae)
r2_overall = np.mean(r2)

# Print results in table format
try:
    from tabulate import tabulate
    table = [
        ["Calories", mae[0], rmse[0], nmae[0], nrmse[0], r2[0]],
        ["Mass", mae[1], rmse[1], nmae[1], nrmse[1], r2[1]],
        ["Fat", mae[2], rmse[2], nmae[2], nrmse[2], r2[2]],
        ["Carbohydrate", mae[3], rmse[3], nmae[3], nrmse[3], r2[3]],
        ["Protein", mae[4], rmse[4], nmae[4], nrmse[4], r2[4]],
        ["Overall", mae_overall, rmse_overall, nmae_overall, nrmse_overall, r2_overall],
    ]
    headers = ["", "MAE", "RMSE", "NMAE", "NRMSE", "R2"]
    print("\n=== Nutrition + Calorie Prediction Results ===")
    print(tabulate(table, headers=headers, tablefmt="github"))
except ImportError:
    print("tabulate library is not installed. Please install it with 'pip install tabulate'.")
    print("Calories    MAE: {:.4f}, RMSE: {:.4f}, NMAE: {:.4f}, NRMSE: {:.4f}, R2: {:.4f}".format(mae[0], rmse[0], nmae[0], nrmse[0], r2[0]))
    print("Mass        MAE: {:.4f}, RMSE: {:.4f}, NMAE: {:.4f}, NRMSE: {:.4f}, R2: {:.4f}".format(mae[1], rmse[1], nmae[1], nrmse[1], r2[1]))
    print("Fat         MAE: {:.4f}, RMSE: {:.4f}, NMAE: {:.4f}, NRMSE: {:.4f}, R2: {:.4f}".format(mae[2], rmse[2], nmae[2], nrmse[2], r2[2]))
    print("Carbohydrate MAE: {:.4f}, RMSE: {:.4f}, NMAE: {:.4f}, NRMSE: {:.4f}, R2: {:.4f}".format(mae[3], rmse[3], nmae[3], nrmse[3], r2[3]))
    print("Protein     MAE: {:.4f}, RMSE: {:.4f}, NMAE: {:.4f}, NRMSE: {:.4f}, R2: {:.4f}".format(mae[4], rmse[4], nmae[4], nrmse[4], r2[4]))
    print("Overall     MAE: {:.4f}, RMSE: {:.4f}, NMAE: {:.4f}, NRMSE: {:.4f}, R2: {:.4f}".format(mae_overall, rmse_overall, nmae_overall, nrmse_overall, r2_overall))

# 9. Specific dish_id prediction comparison
def predict_dish(dish_id):
    row = meta[meta['dish_id'] == dish_id]
    if len(row) == 0:
        print(f'Dish {dish_id} not found!')
        return
    img = load_img(row['img_path'].values[0])
    x = preprocess_input(np.expand_dims(img, 0))
    pred = model.predict(x)
    print(f"\nDish {dish_id} prediction process:")
    print(f"Normalized prediction: {pred[0]}")
    pred_inv = scaler.inverse_transform(pred)[0]
    print(f"Restored prediction: {pred_inv}")
    true = row[['calories', 'weight', 'fat', 'carbs', 'protein']].values[0]
    print(f'True value: {true}')

print('\n[Transfer learning model] Dish dish_1565033189:')
predict_dish('dish_1565033189')
print('\n[Transfer learning model] Dish dish_1556572657:')
predict_dish('dish_1556572657')

# Save model and metadata
meta.to_csv('merged_metadata.csv', index=False)
model.save('transfer_model.h5') 
import numpy as np
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Load metadata
meta = pd.read_csv('merged_metadata.csv')
IMG_SIZE = 224

def load_img(path):
    img = Image.open(path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    return np.array(img)

# Create scaler for 5-value normalization (calories, mass, fat, carb, protein)
y_all = meta[['calories', 'weight', 'fat', 'carbs', 'protein']].values.astype(np.float32)
scaler = StandardScaler().fit(y_all)

# Load both models
transfer_model = tf.keras.models.load_model('transfer_model.h5', compile=False)
plain_model = tf.keras.models.load_model('plain_model.h5', compile=False)

# Error table print function
def print_comparison_table(results, y_range):
    comp_names = ['Calories', 'Mass (g)', 'Fat (g)', 'Carbs (g)', 'Protein (g)']
    table = []
    for i in range(0, len(results), 2):
        dish_id = results[i][0]
        pred = results[i][2:7]
        true = results[i+1][2:7]
        for j, comp in enumerate(comp_names):
            actual = float(true[j])
            predicted = float(pred[j])
            abs_error = abs(predicted - actual)
            sq_error = (predicted - actual) ** 2
            nmae = abs_error / y_range[j] if y_range[j] != 0 else float('nan')
            nrmse = (sq_error ** 0.5) / y_range[j] if y_range[j] != 0 else float('nan')
            table.append([dish_id, comp, f"{actual:.2f}", f"{predicted:.2f}", f"{abs_error:.2f}", f"{nmae:.4f}", f"{nrmse:.4f}"])
    headers = ['Dish ID', 'Component', 'Actual', 'Predicted', 'Abs Error', 'NMAE', 'NRMSE']
    try:
        from tabulate import tabulate
        print('\n=== Transfer Learning Model: Dish Prediction Error Table (Original Units, NMAE/NRMSE) ===')
        print(tabulate(table, headers=headers, tablefmt='github'))
    except ImportError:
        print(headers)
        for row in table:
            print(row)

# Prediction and comparison function
def compare_dishes(dish_ids):
    results = []
    y_all = meta[['calories', 'weight', 'fat', 'carbs', 'protein']].values.astype(np.float32)
    scaler = StandardScaler().fit(y_all)
    y_range = np.max(y_all, axis=0) - np.min(y_all, axis=0)
    
    for dish_id in dish_ids:
        row = meta[meta['dish_id'] == dish_id]
        if len(row) == 0:
            print(f'Dish {dish_id} not found!')
            continue
        img = load_img(row['img_path'].values[0])
        x = preprocess_input(np.expand_dims(img, 0))
        true = row[['calories', 'weight', 'fat', 'carbs', 'protein']].values[0]
        
        # Transfer model prediction
        pred_transfer = transfer_model.predict(x)
        pred_transfer_inv = scaler.inverse_transform(pred_transfer)[0]
        
        # Save results
        results.append([dish_id, 'Transfer', *pred_transfer_inv])
        results.append([dish_id, 'True', *true])
    
    print_comparison_table(results, y_range)

# Compare specific dish IDs
dish_ids = ['dish_1556572657', 'dish_1565033189']
compare_dishes(dish_ids)
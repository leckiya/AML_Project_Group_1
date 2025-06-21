import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import shutil


def process_metadata(metadata_file, output_file):
    """
    Process metadata file to extract dish information in a clean format
    
    Args:
        metadata_file (str): Path to the raw metadata file
        output_file (str): Path to save the processed metadata
    """
    print(f"Processing metadata from {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        raw_lines = f.readlines()
    
    processed_data = []
    
    for line in tqdm(raw_lines, desc="Processing metadata"):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split(',')
        if len(parts) < 6:
            continue
            
        dish_id = parts[0].strip()
        
        try:
            nutrients = [
                float(parts[1]),  # mass
                float(parts[2]),  # calories
                float(parts[3]),  # fat
                float(parts[4]),  # carbs
                float(parts[5])   # protein
            ]
            
            processed_data.append([dish_id] + nutrients)
            
        except (ValueError, IndexError) as e:
            print(f"Error processing line for dish {dish_id}: {e}")
            continue
    
    columns = ['dish_id', 'mass', 'calories', 'fat', 'carbs', 'protein']
    df = pd.DataFrame(processed_data, columns=columns)
    
    df.to_csv(output_file, index=False)
    print(f"Processed metadata saved to {output_file}")
    print(f"Total dishes processed: {len(df)}")
    
    return df


def verify_image_paths(df, imagery_dir):
    """
    Verify that images exist for all dishes in the DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame with dish IDs
        imagery_dir (str): Path to the imagery directory
        
    Returns:
        pd.DataFrame: DataFrame with only valid dishes
    """
    print("Verifying image paths...")
    
    valid_dishes = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
        dish_id = row['dish_id']
        img_path = os.path.join(imagery_dir, 'realsense_overhead', dish_id, 'rgb.png')
        
        if os.path.exists(img_path):
            valid_dishes.append(row)
    
    valid_df = pd.DataFrame(valid_dishes)
    print(f"Valid dishes with images: {len(valid_df)} out of {len(df)}")
    
    return valid_df


def merge_metadata(cafe1_file, cafe2_file, output_file):
    """
    Merge metadata from both cafeterias
    
    Args:
        cafe1_file (str): Path to cafe1 metadata
        cafe2_file (str): Path to cafe2 metadata
        output_file (str): Path to save the merged metadata
    """
    print("Merging metadata from both cafeterias...")
    
    cafe1_df = pd.read_csv(cafe1_file)
    cafe2_df = pd.read_csv(cafe2_file)
    
    merged_df = pd.concat([cafe1_df, cafe2_df], ignore_index=True)
    
    merged_df = merged_df.drop_duplicates(subset=['dish_id'])
    
    merged_df.to_csv(output_file, index=False)
    print(f"Merged metadata saved to {output_file}")
    print(f"Total dishes in merged file: {len(merged_df)}")
    
    return merged_df


def create_train_val_splits(dish_ids, split_ratio=0.8, output_dir='splits', seed=42):
    """
    Create train/validation splits
    
    Args:
        dish_ids (list): List of dish IDs
        split_ratio (float): Ratio of train to total
        output_dir (str): Directory to save the splits
        seed (int): Random seed for reproducibility
    """
    np.random.seed(seed)
    
    dish_ids = np.array(dish_ids)
    np.random.shuffle(dish_ids)
    
    train_size = int(len(dish_ids) * split_ratio)
    train_ids = dish_ids[:train_size]
    val_ids = dish_ids[train_size:]
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'train_ids.txt'), 'w') as f:
        for dish_id in train_ids:
            f.write(f"{dish_id}\n")
    
    with open(os.path.join(output_dir, 'val_ids.txt'), 'w') as f:
        for dish_id in val_ids:
            f.write(f"{dish_id}\n")
            
    print(f"Created splits: {len(train_ids)} train, {len(val_ids)} validation")


def main():
    parser = argparse.ArgumentParser(description='Preprocess Nutrition5k dataset')
    parser.add_argument('--data_dir', type=str, default='nutrition5k_dataset',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help='Directory to save processed data')
    parser.add_argument('--cafe1_only', action='store_true',
                        help='Use only cafe1 data')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    cafe1_metadata = os.path.join(args.data_dir, 'metadata/dish_metadata_cafe1.csv')
    cafe1_processed = os.path.join(args.output_dir, 'cafe1_processed.csv')
    cafe1_df = process_metadata(cafe1_metadata, cafe1_processed)
    
    if not args.cafe1_only:
        cafe2_metadata = os.path.join(args.data_dir, 'metadata/dish_metadata_cafe2.csv')
        cafe2_processed = os.path.join(args.output_dir, 'cafe2_processed.csv')
        cafe2_df = process_metadata(cafe2_metadata, cafe2_processed)
        
        merged_file = os.path.join(args.output_dir, 'merged_metadata.csv')
        merged_df = merge_metadata(cafe1_processed, cafe2_processed, merged_file)
        
        imagery_dir = os.path.join(args.data_dir, 'imagery')
        valid_df = verify_image_paths(merged_df, imagery_dir)
        valid_file = os.path.join(args.output_dir, 'valid_metadata.csv')
        valid_df.to_csv(valid_file, index=False)
        
        splits_dir = os.path.join(args.output_dir, 'splits')
        create_train_val_splits(valid_df['dish_id'].tolist(), output_dir=splits_dir)
    else:
        # Just use cafe1
        imagery_dir = os.path.join(args.data_dir, 'imagery')
        valid_df = verify_image_paths(cafe1_df, imagery_dir)
        valid_file = os.path.join(args.output_dir, 'valid_metadata.csv')
        valid_df.to_csv(valid_file, index=False)
        
        splits_dir = os.path.join(args.output_dir, 'splits')
        create_train_val_splits(valid_df['dish_id'].tolist(), output_dir=splits_dir)
    
    print("Preprocessing complete!")


if __name__ == "__main__":
    main() 
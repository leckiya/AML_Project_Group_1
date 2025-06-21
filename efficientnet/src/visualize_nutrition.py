import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

def load_metadata(metadata_file):
    """
    Load metadata from CSV file
    
    Args:
        metadata_file (str): Path to the metadata file
        
    Returns:
        pd.DataFrame: DataFrame with nutrient information
    """
    # Check if it's a processed file or raw file
    if os.path.exists(metadata_file):
        try:
            # Try loading as processed CSV with headers
            df = pd.read_csv(metadata_file)
            if 'dish_id' in df.columns and 'mass' in df.columns:
                return df
        except:
            pass
            
    # If the file exists but couldn't be loaded as processed data, 
    # assume it's raw and needs processing
    print(f"Processing raw metadata from {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        raw_lines = f.readlines()
    
    processed_data = []
    
    for line in raw_lines:
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
            
        except (ValueError, IndexError):
            continue
    
    columns = ['dish_id', 'mass', 'calories', 'fat', 'carbs', 'protein']
    df = pd.DataFrame(processed_data, columns=columns)
    
    return df

def visualize_distributions(df, output_dir='.'):
    """
    Create visualizations of nutrient distributions
    
    Args:
        df (pd.DataFrame): DataFrame with nutrient information
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 12))
    
    nutrients = ['mass', 'calories', 'fat', 'carbs', 'protein']
    
    for i, nutrient in enumerate(nutrients):
        plt.subplot(2, 3, i+1)
        sns.histplot(df[nutrient], kde=True)
        plt.title(f'Distribution of {nutrient.capitalize()}')
        plt.xlabel(nutrient.capitalize())
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'nutrient_distributions.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(12, 8))
    df_melted = df.melt(id_vars='dish_id', value_vars=nutrients, 
                       var_name='Nutrient', value_name='Value')
    
    sns.boxplot(x='Nutrient', y='Value', data=df_melted)
    plt.title('Boxplots of Nutrient Values')
    plt.ylabel('Value')
    plt.savefig(os.path.join(output_dir, 'nutrient_boxplots.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(15, 15))
    sns.pairplot(df[nutrients])
    plt.savefig(os.path.join(output_dir, 'nutrient_pairplot.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(10, 8))
    corr = df[nutrients].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Between Nutrients')
    plt.savefig(os.path.join(output_dir, 'nutrient_correlation.png'), dpi=300)
    plt.close()
    
    stats = df[nutrients].describe()
    stats.to_csv(os.path.join(output_dir, 'nutrient_statistics.csv'))
    
    print(f"Visualization images saved to {output_dir}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Visualize Nutrition5k ground truth data')
    parser.add_argument('--metadata_file', type=str, 
                        default='processed_data/valid_metadata.csv',
                        help='Path to the metadata file')
    parser.add_argument('--output_dir', type=str, 
                        default='visualizations',
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    df = load_metadata(args.metadata_file)
    print(f"Loaded metadata with {len(df)} dishes")
    
    stats = visualize_distributions(df, args.output_dir)
    
    print("\nSummary Statistics:")
    print(stats)

if __name__ == "__main__":
    main() 
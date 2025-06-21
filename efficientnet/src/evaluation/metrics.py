import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(predictions, targets, output_names=None, normalization='range'):
    """
    Compute regression metrics for predictions vs. targets
    
    Args:
        predictions (np.ndarray): Predicted values [N, num_outputs]
        targets (np.ndarray): Ground truth values [N, num_outputs]
        output_names (list, optional): Names of the outputs for reporting
        normalization (str, optional): Normalization method for RMSE and MAE:
                                      'range': normalize by (max-min)
                                      'mean': normalize by mean
                                      'std': normalize by standard deviation
        
    Returns:
        dict: Dictionary of metrics
    """
    if output_names is None:
        output_names = ['mass', 'calories', 'fat', 'carbs', 'protein']
        
    num_outputs = predictions.shape[1]
    
    overall_mse = mean_squared_error(targets, predictions)
    overall_rmse = np.sqrt(overall_mse)
    overall_mae = mean_absolute_error(targets, predictions)
    overall_r2 = r2_score(targets, predictions)
    
    norm_factors = {}
    for i in range(num_outputs):
        y_true = targets[:, i]
        if normalization == 'range':
            # Normalize by range (max - min)
            norm_factors[i] = np.max(y_true) - np.min(y_true)
        elif normalization == 'mean':
            # Normalize by mean
            norm_factors[i] = np.mean(y_true)
        elif normalization == 'std':
            # Normalize by standard deviation
            norm_factors[i] = np.std(y_true)
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")
    
    per_output_metrics = {}
    overall_nrmse_values = []
    overall_nmae_values = []
    
    for i in range(num_outputs):
        output_name = output_names[i]
        y_true = targets[:, i]
        y_pred = predictions[:, i]
        norm_factor = norm_factors[i]
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        nrmse = rmse / norm_factor if norm_factor != 0 else float('nan')
        nmae = mae / norm_factor if norm_factor != 0 else float('nan')
        
        overall_nrmse_values.append(nrmse)
        overall_nmae_values.append(nmae)
        
        # Mean absolute percentage error (handle potential div by zero)
        mask = y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = float('nan')
        
        per_output_metrics[output_name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'nrmse': nrmse,
            'nmae': nmae,
            'norm_factor': norm_factor
        }
    
    overall_nrmse = np.nanmean(overall_nrmse_values)
    overall_nmae = np.nanmean(overall_nmae_values)
    
    return {
        'overall': {
            'mse': overall_mse,
            'rmse': overall_rmse,
            'mae': overall_mae,
            'r2': overall_r2,
            'nrmse': overall_nrmse,
            'nmae': overall_nmae
        },
        'per_output': per_output_metrics,
        'normalization_method': normalization
    }


def print_metrics(metrics):
    """
    Print metrics in a readable format
    
    Args:
        metrics (dict): Dictionary of metrics from compute_metrics
    """
    norm_method = metrics.get('normalization_method', 'unknown')
    
    print("\n===== Overall Metrics =====")
    print(f"MSE:  {metrics['overall']['mse']:.4f}")
    print(f"RMSE: {metrics['overall']['rmse']:.4f}")
    print(f"MAE:  {metrics['overall']['mae']:.4f}")
    print(f"R²:   {metrics['overall']['r2']:.4f}")
    
    if 'nrmse' in metrics['overall']:
        print(f"NRMSE ({norm_method}): {metrics['overall']['nrmse']:.4f}")
        print(f"NMAE ({norm_method}): {metrics['overall']['nmae']:.4f}")
    
    print("\n===== Per-Output Metrics =====")
    for output_name, output_metrics in metrics['per_output'].items():
        print(f"\n-- {output_name.upper()} --")
        print(f"MSE:  {output_metrics['mse']:.4f}")
        print(f"RMSE: {output_metrics['rmse']:.4f}")
        print(f"MAE:  {output_metrics['mae']:.4f}")
        print(f"R²:   {output_metrics['r2']:.4f}")
        print(f"MAPE: {output_metrics['mape']:.2f}%")
        
        if 'nrmse' in output_metrics:
            print(f"NRMSE ({norm_method}): {output_metrics['nrmse']:.4f}")
            print(f"NMAE ({norm_method}): {output_metrics['nmae']:.4f}")
            print(f"Norm factor: {output_metrics['norm_factor']:.4f}")


def plot_prediction_vs_target(predictions, targets, output_names=None, figsize=(15, 10), save_path=None, normalization='range'):
    """
    Plot predicted vs. target values for each output
    
    Args:
        predictions (np.ndarray): Predicted values [N, num_outputs]
        targets (np.ndarray): Ground truth values [N, num_outputs]
        output_names (list, optional): Names of the outputs
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the figure
        normalization (str, optional): Normalization method ('range', 'mean', or 'std')
        
    Returns:
        matplotlib.figure.Figure: The figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    if output_names is None:
        output_names = ['Mass', 'Calories', 'Fat', 'Carbs', 'Protein']
    
    num_outputs = predictions.shape[1]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for i in range(min(num_outputs, len(axes))):
        ax = axes[i]
        y_true = targets[:, i]
        y_pred = predictions[:, i]
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate normalization factor
        if normalization == 'range':
            norm_factor = np.max(y_true) - np.min(y_true)
        elif normalization == 'mean':
            norm_factor = np.mean(y_true)
        elif normalization == 'std':
            norm_factor = np.std(y_true)
        else:
            raise ValueError(f"Unknown normalization method: {normalization}")
            
        # Calculate normalized RMSE
        nrmse = rmse / norm_factor if norm_factor != 0 else float('nan')
        
        # Create scatter plot
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.5, ax=ax)
        
        # Add perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Add labels and metrics
        ax.set_title(f"{output_names[i]}")
        ax.set_xlabel("Ground Truth")
        ax.set_ylabel("Predicted")
        ax.text(0.05, 0.95, f"RMSE: {rmse:.4f}\nNRMSE ({normalization}): {nrmse:.4f}\nR²: {r2:.4f}", 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Remove unused subplots
    for i in range(num_outputs, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig 
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator

def extract_tensorboard_data(log_dir):
    """
    Extract scalar data from TensorBoard logs
    
    Args:
        log_dir (str): Path to TensorBoard log directory
        
    Returns:
        dict: Dictionary containing extracted data
    """
    event_acc = event_accumulator.EventAccumulator(log_dir)
    event_acc.Reload()
    
    tags = event_acc.Tags()['scalars']
    
    data = {}
    for tag in tags:
        events = event_acc.Scalars(tag)
        data[tag] = {
            'step': [event.step for event in events],
            'value': [event.value for event in events]
        }
    
    return data

def plot_training_metrics(data, output_path=None, title=None):
    """
    Plot training metrics from TensorBoard data
    
    Args:
        data (dict): TensorBoard data
        output_path (str, optional): Path to save the plot
        title (str, optional): Title for the plot
        
    Returns:
        matplotlib.figure.Figure: Figure with the plot
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if 'Loss/train' in data:
        steps = data['Loss/train']['step']
        values = data['Loss/train']['value']
        ax.plot(steps, values, 'b-', label='Training Loss')
    
    if 'Loss/val' in data:
        steps = data['Loss/val']['step']
        values = data['Loss/val']['value']
        ax.plot(steps, values, 'r-', label='Validation Loss')
    
    if 'LR' in data:
        ax2 = ax.twinx()
        steps = data['LR']['step']
        values = data['LR']['value']
        ax2.plot(steps, values, 'g-', label='Learning Rate')
        ax2.set_ylabel('Learning Rate', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title or 'Training and Validation Loss')
    ax.legend(loc='upper left')
    
    if 'LR' in data:
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize training metrics from TensorBoard logs')
    parser.add_argument('--log_dir', type=str, required=True, help='Path to TensorBoard log directory')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the visualization')
    parser.add_argument('--title', type=str, default=None, help='Title for the plot')
    
    args = parser.parse_args()
    
    data = extract_tensorboard_data(args.log_dir)
    
    if args.output_path is None:
        log_dir_name = os.path.basename(os.path.normpath(args.log_dir))
        args.output_path = os.path.join('visualizations', f'{log_dir_name}_training_metrics.png')
        os.makedirs('visualizations', exist_ok=True)
    
    fig = plot_training_metrics(data, args.output_path, args.title)
    
    if args.output_path is None:
        plt.show()

if __name__ == "__main__":
    main() 
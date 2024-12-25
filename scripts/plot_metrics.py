# scripts/plot_metrics.py

import argparse
import json
import matplotlib.pyplot as plt
import os
import yaml
import logging

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def plot_metrics(metrics_path, output_path=None):
    logging.info(f"Loading metrics from {metrics_path}")
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    epochs = range(1, len(metrics['train_loss']) + 1)
    
    plt.figure(figsize=(10, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['train_loss'], label='Train Loss')
    plt.plot(epochs, metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics['train_acc'], label='Train Accuracy')
    plt.plot(epochs, metrics['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
        logging.info(f"Saved plot to {output_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot Training Metrics')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--output', type=str, help='Path to save the plot image')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting plot_metrics script")
    
    config = load_yaml(args.config)
    experiment_name = config['experiment']['name']
    metrics_path = os.path.join(config['logging']['log_dir'], experiment_name, 'metrics.json')
    
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    plot_metrics(metrics_path, args.output)

if __name__ == "__main__":
    main()
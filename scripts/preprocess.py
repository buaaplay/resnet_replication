# evaluate.py

import argparse
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
import os
import yaml
import logging
from utils import load_yaml, setup_logging, load_checkpoint

def get_dataloader(processed_dir, batch_size, split='test'):
    data_path = os.path.join(processed_dir, split, 'data.pt')
    data, labels = torch.load(data_path, weights_only=True)  # 设置 weights_only=True
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def evaluate_model(config):
    setup_logging(config['logging']['log_dir'])
    logging.info("Starting evaluation process")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Data loader
    test_loader = get_dataloader(config['dataset']['path'], config['evaluation']['batch_size'], 'test')
    
    # Model
    if config['model']['architecture'] == 'resnet50':
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config['dataset']['num_classes'])
    else:
        raise ValueError("Unsupported architecture")
    model = model.to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(config['logging']['checkpoint_dir'], 'best_model.pt')
    model, epoch, loss = load_checkpoint(checkpoint_path, model)
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Evaluation
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    loss = running_loss / total
    acc = correct / total
    logging.info(f'Test Loss: {loss:.4f} - Test Accuracy: {acc:.4f}')
    return loss, acc

def main():
    parser = argparse.ArgumentParser(description='Evaluate ResNet Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    config = load_yaml(args.config)
    evaluate_model(config)

if __name__ == "__main__":
    main()
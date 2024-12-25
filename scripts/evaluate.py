# scripts/evaluate.py

import argparse
import torch
import torch.nn as nn
import os
import yaml
import logging
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models
from scripts.utils import load_yaml, setup_logging, load_checkpoint

def get_dataloader(processed_dir, batch_size, split='test'):
    """
    根据指定的 split('train'/'test') 加载数据。默认使用 'test'。
    """
    data_path = os.path.join(processed_dir, split, 'data.pt')
    data, labels = torch.load(data_path)
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def evaluate_model(config, checkpoint_path=None):
    """
    加载指定的checkpoint进行评估，如果checkpoint_path=None，
    则自动使用best模型或最新的模型。
    """
    # 获取实验名称
    experiment_name = config['experiment'].get('name', 'default_exp')
    
    # 设置日志目录
    log_dir = os.path.join(config['logging']['log_dir'], experiment_name)
    setup_logging(log_dir)
    logging.info("Starting evaluation process")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # 数据加载器
    test_loader = get_dataloader(
        config['dataset']['path'],
        config['evaluation']['batch_size'],
        split='test'
    )
    
    # 创建模型
    if config['model']['architecture'] == 'resnet50':
        weights = 'DEFAULT' if config['model']['pretrained'] else None
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config['dataset']['num_classes'])
    else:
        raise ValueError(f"Unsupported architecture {config['model']['architecture']}")
    model = model.to(device)
    
    # 如果没有指定 checkpoint_path，则默认加载 best 或最新
    if checkpoint_path is None:
        # 假设有一个 best 模型
        # 例如: models/checkpoints/experiment2/checkpoint_epoch_best.pth
        checkpoint_dir = os.path.join(config['logging']['checkpoint_dir'], experiment_name)
        best_ckpt = os.path.join(checkpoint_dir, 'checkpoint_epoch_best.pth')
        if os.path.exists(best_ckpt):
            checkpoint_path = best_ckpt
            logging.info(f"Using best checkpoint: {checkpoint_path}")
        else:
            # 若没有 best，就用最后一个 epoch
            logging.warning("No best checkpoint found, using the last epoch checkpoint.")
            # 简单做法：找到所有 checkpoint 文件，选 epoch 最大的
            ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')]
            if not ckpt_files:
                raise FileNotFoundError("No checkpoint found in the checkpoint directory.")
            # 例如 checkpoint_epoch_22.pth -> epoch=22
            epochs = [int(f.split('_')[-1].split('.')[0]) for f in ckpt_files if f.split('_')[-1].startswith('best')==False]
            last_epoch = max(epochs)
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{last_epoch}.pth')
            logging.info(f"Using the last epoch checkpoint: {checkpoint_path}")
    
    # 加载模型权重
    model, _, epoch, loss = load_checkpoint(checkpoint_path, model, optimizer=None)
    logging.info(f"Loaded checkpoint from {checkpoint_path}")
    logging.info(f"Checkpoint info - epoch: {epoch}, loss: {loss}")
    
    # 开始评估
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / total
    acc = correct / total
    logging.info(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {acc:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate ResNet Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a specific checkpoint file (if not provided, use best or last)')
    args = parser.parse_args()
    
    config = load_yaml(args.config)
    evaluate_model(config, args.checkpoint)

if __name__ == "__main__":
    main()

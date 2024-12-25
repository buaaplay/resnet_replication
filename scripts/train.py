# scripts/train.py

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
from torch.utils.data import DataLoader, TensorDataset
import os
import yaml
import logging
from scripts.utils import load_yaml, setup_logging, save_checkpoint, EarlyStopping  
from torch.utils.tensorboard import SummaryWriter
import json

def get_dataloader(processed_dir, batch_size, split='train'):
    data_path = os.path.join(processed_dir, split, 'data.pt')
    data, labels = torch.load(data_path)
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
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
    return loss, acc

def train_model(config):
    # 获取实验名称
    experiment_name = config['experiment']['name']
    
    # 设置日志和检查点的目录
    log_dir = os.path.join(config['logging']['log_dir'], experiment_name)
    checkpoint_dir = os.path.join(config['logging']['checkpoint_dir'], experiment_name)
    
    setup_logging(log_dir)
    logging.info(f"Starting training for experiment: {experiment_name}")
    
    # 强制转换参数类型
    weight_decay = float(config['training']['weight_decay'])
    logging.info(f"Type of weight_decay after conversion: {type(weight_decay)}")  # 应该显示 <class 'float'>
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Data loaders
    train_loader = get_dataloader(config['dataset']['path'], config['training']['batch_size'], 'train')
    val_loader = get_dataloader(config['dataset']['path'], config['evaluation']['batch_size'], 'test')
    
    # Model
    if config['model']['architecture'] == 'resnet50':
        weights = 'DEFAULT' if config['model']['pretrained'] else None
        model = models.resnet50(weights=weights)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config['dataset']['num_classes'])
    else:
        raise ValueError("Unsupported architecture")
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_name = config['training']['optimizer']
    
    # 获取优化器参数
    optimizer_params = {
        'lr': config['training']['learning_rate'],
        'weight_decay': weight_decay
    }
    
    # 如果优化器支持 momentum，则添加
    if optimizer_name in ['SGD', 'RMSprop']:
        momentum = float(config['training'].get('momentum', 0.0))
        optimizer_params['momentum'] = momentum
    
    try:
        optimizer = getattr(optim, optimizer_name)(model.parameters(), **optimizer_params)
    except AttributeError:
        raise ValueError(f"Optimizer '{optimizer_name}' is not recognized.")
    
    # Scheduler
    if config['training']['scheduler'] == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['training']['step_size'], gamma=config['training']['gamma'])
    else:
        scheduler = None
    
    # 初始化 Early Stopping
    early_stopping = EarlyStopping(patience=config['training']['early_stopping']['patience'],
                                   delta=config['training']['early_stopping']['delta'])
    
    # 初始化 TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    
    # 初始化记录指标
    metrics = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0  # 用于保存最佳模型
    
    # Training loop
    try:
        for epoch in range(1, config['training']['num_epochs'] + 1):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss / total
            train_acc = correct / total
            logging.info(f'Epoch {epoch}/{config["training"]["num_epochs"]} - Loss: {train_loss:.4f} - Acc: {train_acc:.4f}')
            
            # 记录到 TensorBoard
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            
            # 评估验证集
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
            logging.info(f'Validation - Loss: {val_loss:.4f} - Acc: {val_acc:.4f}')
            
            # 记录到 TensorBoard
            writer.add_scalar('Loss/Validation', val_loss, epoch)
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            
            # 记录指标
            metrics['train_loss'].append(train_loss)
            metrics['train_acc'].append(train_acc)
            metrics['val_loss'].append(val_loss)
            metrics['val_acc'].append(val_acc)
            
            # 更新学习率调度器
            if scheduler:
                scheduler.step()
            
            # 早停检查
            early_stopping(val_loss)
            if early_stopping.early_stop:
                logging.info("Early stopping triggered. Stopping training.")
                break
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss,
                }
                save_checkpoint(best_checkpoint, checkpoint_dir, 'best')
                logging.info(f'Best model updated at epoch {epoch} with Val Acc: {val_acc:.4f}')
            
            # 保存检查点
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }
            save_checkpoint(checkpoint, checkpoint_dir, epoch)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user. Saving current model state.")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
        }
        save_checkpoint(checkpoint, checkpoint_dir, epoch)
        logging.info("Checkpoint saved. Exiting.")
    
    # 保存指标到 JSON 文件
    metrics_path = os.path.join(log_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    logging.info(f"Metrics saved to {metrics_path}")
    
    logging.info("Training completed")
    writer.close()

def main():
    parser = argparse.ArgumentParser(description='Train ResNet Model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    config = load_yaml(args.config)
    train_model(config)

if __name__ == "__main__":
    main()

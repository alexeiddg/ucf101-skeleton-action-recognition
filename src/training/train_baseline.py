import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.datasets.skeleton_dataset import SkeletonDataset
from src.models.baseline_mlp import create_baseline_model
from src.utils.seed import set_seed
from src.utils.metrics import compute_metrics
from src.utils.plotting import plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description='Train baseline MLP model')
    parser.add_argument('--config', type=str, default='configs/baseline.yaml',
                        help='Path to config file')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []
    
    for skeletons, labels in dataloader:
        skeletons = skeletons.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(skeletons)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track loss and predictions
        epoch_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_metrics(all_labels, all_preds)
    
    return epoch_loss / len(dataloader), metrics['accuracy']


def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for skeletons, labels in dataloader:
            skeletons = skeletons.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(skeletons)
            loss = criterion(outputs, labels)
            
            # Track loss and predictions
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_metrics(all_labels, all_preds)
    
    return val_loss / len(dataloader), metrics['accuracy'], metrics


def main():
    args = parse_args()
    
    config = load_config(args.config)
    
    set_seed(config['seed'])
    
    os.makedirs(config['paths']['save_dir'], exist_ok=True)
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = SkeletonDataset(
        data_path=config['dataset']['data_path'],
        split='train',
        num_frames=config['dataset']['num_frames'],
        num_joints=config['dataset']['num_joints'],
        num_coords=config['dataset']['num_coords'],
        class_names=config['dataset']['class_names']
    )
    
    val_dataset = SkeletonDataset(
        data_path=config['dataset']['data_path'],
        split='val',
        num_frames=config['dataset']['num_frames'],
        num_joints=config['dataset']['num_joints'],
        num_coords=config['dataset']['num_coords'],
        class_names=config['dataset']['class_names']
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model = create_baseline_model(config['model'])
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['training']['scheduler_factor'],
        patience=config['training']['scheduler_patience'],
        min_lr=config['training']['scheduler_min_lr'],
        verbose=True
    )
    
    writer = SummaryWriter(config['paths']['log_dir'])
    
    best_val_accuracy = 0.0
    early_stopping_counter = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    print("Starting training...")
    for epoch in range(config['training']['num_epochs']):
        train_loss, train_accuracy = train_epoch(model, train_dataloader, criterion, optimizer, device)
        
        val_loss, val_accuracy, val_metrics = validate(model, val_dataloader, criterion, device)
        
        scheduler.step(val_loss)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.add_scalar('Precision/val', val_metrics['precision'], epoch)
        writer.add_scalar('Recall/val', val_metrics['recall'], epoch)
        writer.add_scalar('F1/val', val_metrics['f1'], epoch)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{config['training']['num_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_metrics': val_metrics
            }, os.path.join(config['paths']['save_dir'], 'best_model.pth'))
            print(f"Saved best model with validation accuracy: {val_accuracy:.4f}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= config['training']['early_stopping_patience']:
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_accuracy,
        'val_metrics': val_metrics
    }, os.path.join(config['paths']['save_dir'], 'final_model.pth'))
    
    plot_training_curves(
        train_losses,
        val_losses,
        train_accuracies,
        val_accuracies,
        save_path=os.path.join(config['paths']['save_dir'], 'training_curves.png')
    )
    
    writer.close()
    
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")


if __name__ == '__main__':
    main()
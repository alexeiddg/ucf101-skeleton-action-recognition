import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.datasets.skeleton_dataset import SkeletonDataset
from src.models.baseline_mlp import create_baseline_model
from src.models.cnn_lstm import create_cnn_lstm_model
from src.utils.seed import set_seed
from src.utils.metrics import compute_metrics, compute_confusion_matrix
from src.utils.plotting import plot_confusion_matrix
from src.utils.ucf101_classes import UCF101_CLASSES


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config, checkpoint_path, device):
    if config['model']['name'] == 'BaselineMLP':
        model = create_baseline_model(config['model'])
    elif config['model']['name'] == 'CNNLSTMModel':
        model = create_cnn_lstm_model(config['model'])
    else:
        raise ValueError(f"Unknown model name: {config['model']['name']}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def evaluate(model, dataloader, device):
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for skeletons, labels in dataloader:
            skeletons = skeletons.to(device)

            # Forward pass
            outputs = model(skeletons)
            probs = torch.softmax(outputs, dim=1)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def main():
    args = parse_args()

    config = load_config(args.config)
    set_seed(config['seed'])

    if args.output_dir is None:
        output_dir = os.path.join(config['paths']['save_dir'], 'evaluation')
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    test_dataset = SkeletonDataset(
        data_path=config['dataset']['data_path'],
        split='test',
        num_frames=config['dataset']['num_frames'],
        num_joints=config['dataset']['num_joints'],
        num_coords=config['dataset']['num_coords'],
        class_names=config['dataset']['class_names']
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    model = load_model(config, args.checkpoint, device)
    
    print("Evaluating model...")
    preds, labels, probs = evaluate(model, test_dataloader, device)
    
    metrics = compute_metrics(labels, preds)
    
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
    
    cm = compute_confusion_matrix(labels, preds)
    
    if config['dataset']['class_names'] is not None:
        class_names = config['dataset']['class_names']
    else:
        unique_labels = np.unique(np.concatenate([labels, preds]))
        class_names = [UCF101_CLASSES[i] for i in unique_labels]
    
    plot_confusion_matrix(
        cm,
        class_names,
        normalize=True,
        title='Normalized Confusion Matrix',
        save_path=os.path.join(output_dir, 'confusion_matrix.png')
    )
    
    print(f"Evaluation results saved to {output_dir}")


if __name__ == '__main__':
    main()
import os
import sys
import yaml
import argparse
import numpy as np
import torch
import torch.nn as nn
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.baseline_mlp import create_baseline_model
from src.models.cnn_lstm import create_cnn_lstm_model
from src.utils.seed import set_seed
from src.utils.ucf101_classes import get_class_name


def parse_args():
    parser = argparse.ArgumentParser(description='Predict action class for a sample')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--sample_id', type=str, required=True,
                        help='ID of the sample to predict')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to the dataset file (overrides config)')
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


def load_sample(data_path, sample_id, config):
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    sample = None
    for annotation in data['annotations']:
        if annotation['id'] == sample_id:
            sample = annotation
            break
    
    if sample is None:
        raise ValueError(f"Sample with ID {sample_id} not found in the dataset")

    keypoints = sample['keypoint']
    label = sample['label']

    if keypoints.shape[0] > 0:
        keypoints = keypoints[0]
    else:
        keypoints = np.zeros((1, config['dataset']['num_joints'], 3))

    processed_keypoints = process_keypoints(
        keypoints,
        config['dataset']['num_frames'],
        config['dataset']['num_joints'],
        config['dataset']['num_coords']
    )

    skeleton_tensor = torch.tensor(processed_keypoints, dtype=torch.float32).unsqueeze(0)
    
    return skeleton_tensor, label


def process_keypoints(keypoints, num_frames, num_joints, num_coords):
    num_frames_data = keypoints.shape[0]
    
    if num_frames_data >= num_frames:
        keypoints = keypoints[:num_frames]
    else:
        padding = np.zeros((num_frames - num_frames_data, keypoints.shape[1], keypoints.shape[2]))
        keypoints = np.concatenate([keypoints, padding], axis=0)
    
    keypoints = keypoints[:, :, :num_coords]

    min_vals = np.min(keypoints, axis=(0, 1))
    max_vals = np.max(keypoints, axis=(0, 1))

    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1

    keypoints = (keypoints - min_vals) / range_vals
    keypoints = keypoints.reshape(num_frames, -1)
    
    return keypoints


def predict(model, skeleton_tensor, device):
    skeleton_tensor = skeleton_tensor.to(device)
    
    with torch.no_grad():
        # Forward pass
        outputs = model(skeleton_tensor)
        probs = torch.softmax(outputs, dim=1)
        
        # Get prediction
        _, pred = torch.max(outputs, 1)
    
    return pred.item(), probs[0].cpu().numpy()


def main():
    args = parse_args()

    config = load_config(args.config)

    set_seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(config, args.checkpoint, device)

    data_path = args.data_path if args.data_path is not None else config['dataset']['data_path']

    print(f"Loading sample with ID: {args.sample_id}")
    skeleton_tensor, true_label = load_sample(data_path, args.sample_id, config)

    print("Predicting...")
    pred_label, probs = predict(model, skeleton_tensor, device)

    true_class = get_class_name(true_label)
    pred_class = get_class_name(pred_label)
    
    print("\nPrediction Results:")
    print(f"Sample ID: {args.sample_id}")
    print(f"True Class: {true_class} (Label: {true_label})")
    print(f"Predicted Class: {pred_class} (Label: {pred_label})")
    print(f"Confidence: {probs[pred_label]:.4f}")
    
    top5_indices = np.argsort(probs)[-5:][::-1]
    print("\nTop 5 Predictions:")
    for i, idx in enumerate(top5_indices):
        print(f"{i+1}. {get_class_name(idx)} (Label: {idx}): {probs[idx]:.4f}")


if __name__ == '__main__':
    main()
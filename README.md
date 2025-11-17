# UCF101 Skeleton-Based Action Recognition

This repository contains a deep learning pipeline for human action recognition using 2D skeleton data from the UCF101 dataset. The project implements two models: a baseline MLP and a more sophisticated CNN+LSTM hybrid model.

> Please Revise 'ucf101_skeleton_action_recognition.ipynb' notebook to see full run and breakdown of both model's performance and metrics + a full report on thier performance for this dataset

## Project Structure

```
.
├── configs/               # Configuration files
│   ├── baseline.yaml      # Configuration for baseline MLP model
│   └── cnn_lstm.yaml      # Configuration for CNN+LSTM model
├── data/                  # Dataset directory
│   └── ucf101_2d.pkl      # UCF101 skeleton dataset (not included)
├── src/                   # Source code
│   ├── datasets/          # Dataset loaders
│   ├── models/            # Model definitions
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation and prediction scripts
│   └── utils/             # Utility functions
├── results/               # Training results and model checkpoints
├── README.md              # This file
├── ucf101_skeleton_action_recognition.ipynb   <--- Both models comparative & report
└── requirements.txt       # Project dependencies
```

## Dataset

The project uses the UCF101 dataset with 2D skeleton annotations in the MMAction2 format. The dataset is stored as a pickle file with the following structure:

- `split`: Dictionary containing train, validation, and test identifiers
- `annotations`: List of dictionaries, each containing:
  - `id`: Video clip identifier
  - `frame_num`: Number of frames
  - `img_shape`: Original image shape
  - `label`: Action class (integer)
  - `keypoint`: Skeleton keypoints array with shape [M, T, V, C]
    - M: Number of persons
    - T: Number of frames
    - V: Number of joints
    - C: Coordinates (x, y, score)

## Models

1. **Baseline MLP**: A simple multi-layer perceptron that processes flattened skeleton sequences.
2. **CNN+LSTM**: A hybrid model that uses CNN layers to extract spatial features from skeleton data and BiLSTM layers to capture temporal dynamics.

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Training

To train the baseline MLP model:

```bash
python src/training/train_baseline.py
```

To train the CNN+LSTM model:

```bash
python src/training/train_cnn_lstm.py
```

### Evaluation

To evaluate a trained model:

```bash
python src/evaluation/evaluate.py --config configs/baseline.yaml --checkpoint results/baseline/best_model.pth
```

### Prediction

To predict the action class for a single sample:

```bash
python src/evaluation/predict.py --config configs/baseline.yaml --checkpoint results/baseline/best_model.pth --sample_id <sample_id>
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
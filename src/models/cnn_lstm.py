"""
CNN+LSTM hybrid model for skeleton-based action recognition.
"""

import torch
import torch.nn as nn


class CNNLSTMModel(nn.Module):
    def __init__(self, input_channels, input_frames, input_joints, conv_channels, 
                 conv_kernel_sizes, lstm_hidden_size, lstm_num_layers, lstm_dropout,
                 fc_sizes, fc_dropout, num_classes, bidirectional=True):

        super(CNNLSTMModel, self).__init__()
        
        self.input_channels = input_channels
        self.input_frames = input_frames
        self.input_joints = input_joints
        self.bidirectional = bidirectional
        
        # CNN layers for spatial feature extraction
        cnn_layers = []
        in_channels = input_channels
        
        for i, (out_channels, kernel_size) in enumerate(zip(conv_channels, conv_kernel_sizes)):
            cnn_layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            cnn_layers.append(nn.BatchNorm1d(out_channels))
            cnn_layers.append(nn.ReLU())
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # LSTM layers for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=in_channels * input_joints,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Fully connected layers for classification
        fc_layers = []
        lstm_output_size = lstm_hidden_size * 2 if bidirectional else lstm_hidden_size
        in_features = lstm_output_size
        
        for out_features in fc_sizes:
            fc_layers.append(nn.Linear(in_features, out_features))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(fc_dropout))
            in_features = out_features
        
        fc_layers.append(nn.Linear(in_features, num_classes))
        
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        batch_size = x.size(0)

        x = x.view(batch_size, self.input_frames, self.input_joints, self.input_channels)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size * self.input_frames, self.input_channels, self.input_joints)
        
        x = self.cnn(x)

        x = x.view(batch_size, self.input_frames, -1)
        x, _ = self.lstm(x)
        
        if self.bidirectional:
            x = x[:, -1, :]
        else:
            x = x[:, -1, :]
        
        x = self.fc(x)
        
        return x


def create_cnn_lstm_model(config):
    return CNNLSTMModel(
        input_channels=config['input_channels'],
        input_frames=config['input_frames'],
        input_joints=config['input_joints'],
        conv_channels=config['conv_channels'],
        conv_kernel_sizes=config['conv_kernel_sizes'],
        lstm_hidden_size=config['lstm_hidden_size'],
        lstm_num_layers=config['lstm_num_layers'],
        lstm_dropout=config['lstm_dropout'],
        fc_sizes=config['fc_sizes'],
        fc_dropout=config['fc_dropout'],
        num_classes=config['num_classes'],
        bidirectional=config['bidirectional']
    )
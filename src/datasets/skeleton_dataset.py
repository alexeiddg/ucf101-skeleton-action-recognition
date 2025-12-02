import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class SkeletonDataset(Dataset):
    """
    UCF101 Skeleton Dataset Loader (MMAction2 format)

    annotation keys:
        - frame_dir: video identifier
        - total_frames: number of frames
        - img_shape: (H, W)
        - original_shape: (H, W)
        - label: int
        - keypoint: [M, T, V, C]
        - keypoint_score: [M, T, V]

    valid splits:
        train1, train2, train3
        test1,  test2,  test3

    Args:
        return_valid_mask (bool): when True, also returns a mask marking real (non-padded) frames.
    """

    def __init__(self, data_path, split='train1', num_frames=64,
                 num_joints=17, num_coords=2, class_names=None,
                 return_valid_mask=False):

        self.data_path = data_path
        self.split = split
        self.num_frames = num_frames
        self.num_joints = num_joints
        self.num_coords = num_coords
        self.class_names = class_names
        self.return_valid_mask = return_valid_mask

        with open(self.data_path, 'rb') as f:
            all_data = pickle.load(f)

        self.split_dict = all_data["split"]
        self.annotations = all_data["annotations"]

        if self.split not in self.split_dict:
            raise ValueError(
                f"Invalid split '{self.split}'. "
                f"Available splits: {list(self.split_dict.keys())}"
            )

        split_ids = set(self.split_dict[self.split])
        self.data = []

        for ann in self.annotations:
            if ann["frame_dir"] in split_ids:
                self.data.append(ann)

        if len(self.data) == 0:
            raise RuntimeError(
                f"No samples found for split '{self.split}'. "
                f"Check dataset path or split name."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ann = self.data[idx]

        keypoints = ann["keypoint"]
        label = ann["label"]

        kp = keypoints[0]

        T = kp.shape[0]
        valid_len = min(T, self.num_frames)
        frame_mask = np.zeros(self.num_frames, dtype=np.float32)
        frame_mask[:valid_len] = 1.0

        if T >= self.num_frames:
            kp = kp[:self.num_frames]
        else:
            pad = np.zeros((self.num_frames - T, kp.shape[1], kp.shape[2]))
            kp = np.concatenate([kp, pad], axis=0)

        kp = kp[:, :, :self.num_coords]

        min_vals = kp.min(axis=(0, 1), keepdims=True)
        max_vals = kp.max(axis=(0, 1), keepdims=True)
        denom = (max_vals - min_vals)
        denom[denom == 0] = 1
        kp = (kp - min_vals) / denom

        kp = kp.reshape(self.num_frames, -1)

        if self.return_valid_mask:
            return (
                torch.tensor(kp, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long),
                torch.tensor(frame_mask, dtype=torch.float32),
            )

        return torch.tensor(kp, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

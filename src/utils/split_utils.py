import pickle
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit


def extract_group_id(frame_dir: str) -> str:
    """
    Extract the group identifier from a UCF101 frame_dir string.
    Expected pattern: v_<ClassName>_gXX_cYY -> returns "gXX".
    """
    match = re.search(r"_g(\d+)_", frame_dir)
    if not match:
        raise ValueError(f"Could not parse group id from frame_dir '{frame_dir}'")
    return f"g{match.group(1)}"


def build_grouped_split(
    annotations: Iterable[Dict],
    allowed_ids: Iterable[str],
    n_splits: int = 5,
    fold_idx: int = 0,
    random_state: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Create a train/val split that keeps samples from the same group together.
    Uses StratifiedGroupKFold for balanced class distribution; falls back to
    GroupShuffleSplit if stratification is not feasible.
    """
    allowed_ids = set(allowed_ids)
    filtered = [ann for ann in annotations if ann["frame_dir"] in allowed_ids]

    if len(filtered) == 0:
        raise ValueError("No annotations matched the provided allowed_ids")

    frame_dirs = [ann["frame_dir"] for ann in filtered]
    labels = np.array([ann["label"] for ann in filtered])
    groups = np.array([extract_group_id(ann["frame_dir"]) for ann in filtered])

    # Try stratified split first for better class balance
    try:
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        splits = list(sgkf.split(np.zeros(len(labels)), labels, groups))
        if fold_idx >= len(splits):
            raise IndexError(f"fold_idx {fold_idx} out of range for {len(splits)} splits")
        train_idx, val_idx = splits[fold_idx]
    except Exception:
        # Fallback: group-aware shuffle split (not stratified)
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=1 / n_splits,
            random_state=random_state,
        )
        train_idx, val_idx = next(gss.split(np.zeros(len(labels)), labels, groups))

    train_ids = [frame_dirs[i] for i in train_idx]
    val_ids = [frame_dirs[i] for i in val_idx]
    return train_ids, val_ids


def add_grouped_splits(
    data: Dict,
    base_split: str = "train1",
    n_splits: int = 5,
    fold_idx: int = 0,
    random_state: int = 42,
    train_key: str = None,
    val_key: str = None,
) -> Dict:
    """
    Add grouped train/val splits to an in-memory dataset dictionary.
    Returns the modified dictionary (mutates in place).
    """
    if "split" not in data or "annotations" not in data:
        raise KeyError("Dataset dictionary must contain 'split' and 'annotations' keys")

    if base_split not in data["split"]:
        raise KeyError(f"Base split '{base_split}' not found in dataset splits")

    train_ids, val_ids = build_grouped_split(
        data["annotations"],
        data["split"][base_split],
        n_splits=n_splits,
        fold_idx=fold_idx,
        random_state=random_state,
    )

    train_key = train_key or f"{base_split}_grouped_train"
    val_key = val_key or f"{base_split}_grouped_val"

    data["split"][train_key] = train_ids
    data["split"][val_key] = val_ids
    return data


def save_grouped_splits(
    data_path: str,
    output_path: str = None,
    base_split: str = "train1",
    n_splits: int = 5,
    fold_idx: int = 0,
    random_state: int = 42,
    train_key: str = None,
    val_key: str = None,
) -> str:
    """
    Load a pickle dataset, add grouped splits, and save to disk.
    Returns the path of the saved pickle.
    """
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    add_grouped_splits(
        data,
        base_split=base_split,
        n_splits=n_splits,
        fold_idx=fold_idx,
        random_state=random_state,
        train_key=train_key,
        val_key=val_key,
    )

    output_path = output_path or data_path
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    return output_path

import torch
from torch.utils.data import Dataset


def build_q_from_train_targets(
    train_targets: torch.Tensor,
    catalog_size: int,
) -> torch.Tensor:
    if train_targets.numel() == 0:
        raise ValueError
    train_targets_flat = train_targets.flatten()
    if not (train_targets_flat >= 0).all():
        raise ValueError
    if (train_targets_flat >= catalog_size).any():
        raise ValueError
    counts = torch.bincount(train_targets_flat, minlength=catalog_size)
    return counts.float()


def get_q_counts(yambda_train_dataset: Dataset, yambda_train_len: int, catalog_size: int) -> torch.Tensor:
    train_target_ids = torch.tensor(
        [
            target
            for idx in range(yambda_train_len)
            for target in yambda_train_dataset[idx]["targets"]
        ],
        dtype=torch.long
    )

    q = build_q_from_train_targets(
        train_targets=train_target_ids,
        catalog_size=catalog_size,
    )
    return q
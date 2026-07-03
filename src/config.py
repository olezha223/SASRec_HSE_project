from dataclasses import dataclass

import torch.cuda


@dataclass
class TrainConfig:
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_BATCH_SIZE: int = 128
    EVAL_BATCH_SIZE: int = 128
    NUM_EPOCHS: int = 10
    LEARNING_RATE: float = 1e-3
    TOPK = 100
    CORE_MIN_INTERACTIONS_PER_ITEM = 5
    TEST_INTERVAL_SECONDS = 7 * 24 * 60 * 60
    DATA_DIR = "./data"


train_config = TrainConfig()
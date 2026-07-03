from typing import Any, Dict, List

from torch.utils.data import Dataset


class YambdaEvalDataset(Dataset):
    def __init__(
        self,
        histories: Dict[Any, List[int]],
        targets: Dict[Any, List[int]],
        max_seq_len: int = 100,
    ) -> None:
        super().__init__()
        self.histories = histories
        self.targets = targets
        self.max_seq_len = max_seq_len

        self.samples = []
        for uid, history in histories.items():
            if uid in targets:
                sample = history[-max_seq_len:]
                if len(sample) > 0:
                    self.samples.append(
                        {
                            "uid": uid,
                            "history": sample,
                            "length": len(sample),
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

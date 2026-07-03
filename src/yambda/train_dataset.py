from typing import Dict, Any, List

from torch.utils.data import Dataset


class YambdaTrainDataset(Dataset):
    def __init__(
        self,
        histories: Dict[Any, List[int]],
        max_seq_len: int = 100,
    ) -> None:
        super().__init__()
        self.histories = histories
        self.max_seq_len = max_seq_len

        self.samples = []
        for uid, history in histories.items():
            if len(history) < 2:
                continue
            if len(history) <= max_seq_len:
                self.samples.append(
                    {
                        "uid": uid,
                        "history": history[:-1],
                        "targets": history[1:],
                        "length": len(history) - 1
                    }
                )
            if len(history) > max_seq_len:
                for i in range(0, len(history), max_seq_len):
                    chunk_history = history[i:i+max_seq_len+1]
                    if len(chunk_history) >= 2:
                        self.samples.append(
                            {
                                "uid": uid,
                                "history": chunk_history[:-1],
                                "targets": chunk_history[1:],
                                "length": len(chunk_history) - 1
                            }
                        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

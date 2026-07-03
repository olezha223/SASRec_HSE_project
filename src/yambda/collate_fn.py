from typing import List, Dict, Any

import torch


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    history = []
    lengths = []
    uids = []
    labels = []
    for el in batch:
        history.extend(el['history'])
        lengths.append(el['length'])
        uids.append(el['uid'])
        labels.extend(el.get('targets', []))
    history_tensor = torch.tensor(history, dtype=torch.long)
    length_tensor = torch.tensor(lengths, dtype=torch.long)
    uid_tensor = torch.tensor(uids, dtype=torch.long)
    if labels:
        targets_tensor = torch.tensor(labels, dtype=torch.long)
        return {
            "uid": uid_tensor,
            "length": length_tensor,
            "history": history_tensor,
            "targets": targets_tensor,
        }
    return {
        "uid": uid_tensor,
        "length": length_tensor,
        "history": history_tensor,
    }
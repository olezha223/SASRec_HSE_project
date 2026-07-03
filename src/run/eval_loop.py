from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.metrics import evaluate
from src.utils.to_device import to_device


@torch.no_grad()
def eval_nip(
        dataloader: DataLoader,
        model: torch.nn.Module,
        catalog_size: int,
        topk: int,
        device: str,
        targets,
        evaluate_fn=evaluate
) -> Dict[str, float]:
    model.eval()
    model.to(device)
    candidates = {}

    for batch in tqdm(dataloader):
        batch = to_device(batch, device)

        scores = model(batch)
        topk_items = torch.topk(scores, k=topk, dim=1).indices

        for i, uid in enumerate(batch["uid"]):
            uid = uid.cpu().item()
            candidates[uid] = topk_items[i].cpu().tolist()

    metrics = evaluate_fn(
        targets=targets,
        candidates=candidates,
        catalog_size=catalog_size,
        topk=topk
    )
    return metrics
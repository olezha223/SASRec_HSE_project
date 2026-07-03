from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.to_device import to_device


def train_epoch(
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch_num: int,
    max_epochs: int,
    device: str = "cuda",
) -> tuple[dict[str, float], float]:
    running_loss = 0

    for batch in tqdm(dataloader, desc=f'Эпоха {epoch_num}/{max_epochs}'):
        batch = to_device(batch, device)
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(dataloader)
    return avg_train_loss

def train(
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: str = "cuda",
) -> tuple[dict[str, Any], list[float]]:
    model.train()
    model.to(device)

    train_losses = []

    for epoch in range(num_epochs):
        avg_train_loss = train_epoch(
            dataloader=dataloader,
            model=model,
            optimizer=optimizer,
            device=device,
            epoch_num=epoch,
            max_epochs=num_epochs
        )
        train_losses.append(avg_train_loss)

        print("="*30)
        print(f"Epoch={epoch} | Loss={avg_train_loss}\n")
    print(f"Модель {model.__class__} обучена!")
    return model.state_dict(), train_losses
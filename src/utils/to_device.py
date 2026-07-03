import torch


def to_device(obj, device: torch.device | str):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    elif isinstance(obj, dict):
        return {key: to_device(value, device) for key, value in obj.items()}

    elif isinstance(obj, list):
        return [to_device(item, device) for item in obj]

    elif isinstance(obj, tuple):
        return tuple(to_device(item, device) for item in obj)

    else:
        return obj
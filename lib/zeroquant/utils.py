import torch

def to_device(batch, device):
    """
    Move a batch of data to the specified device.
    Handles nested structures such as dictionaries, lists, tuples, and tensors.
    """
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return type(batch)(to_device(v, device) for v in batch)  # Preserve the type (list/tuple)
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        # If the item is not a tensor or collection, return as is
        return batch

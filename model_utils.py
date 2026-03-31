from pathlib import Path

import torch

from models import resnet26


MODEL_FACTORY = {
    "resnet26": resnet26,
}


def build_model(arch, num_classes=10):
    arch_name = arch.lower()
    if arch_name not in MODEL_FACTORY:
        available = ", ".join(sorted(MODEL_FACTORY))
        raise ValueError(f"Unknown architecture '{arch}'. Available: {available}")
    return MODEL_FACTORY[arch_name](num_classes=num_classes)


def resolve_device(device_name="auto"):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_checkpoint(model, checkpoint_path, map_location="cpu", strict=True):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )
    state_dict = _extract_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=strict)
    return checkpoint if isinstance(checkpoint, dict) else {}


def save_checkpoint(
    checkpoint_path,
    model,
    optimizer=None,
    scheduler=None,
    epoch=None,
    best_acc=None,
    arch="resnet26",
    num_classes=10,
):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": arch,
        "num_classes": num_classes,
        "epoch": epoch,
        "best_acc": best_acc,
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, checkpoint_path)

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import make_splits
from .model import ToyCNN, count_trainable_parameters


@dataclass(frozen=True)
class TrainConfig:
    train_size: int = 1200
    val_size: int = 200
    test_size: int = 120
    batch_size: int = 16
    epochs: int = 20
    eval_every_steps: int = 10
    early_stop_acc: float = 0.95
    lr: float = 3e-3
    weight_decay: float = 1e-3
    seed: int = 7


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def evaluate(model: ToyCNN, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            logits = model(batch["image"])
            loss = F.cross_entropy(logits, batch["label"])
            total_loss += float(loss.item()) * batch["label"].shape[0]
            total_correct += int((logits.argmax(dim=1) == batch["label"]).sum().item())
            total += int(batch["label"].shape[0])
    return {"loss": total_loss / total, "acc": total_correct / total}


def train_model(
    output: Path,
    config: TrainConfig | None = None,
    device_name: str = "auto",
) -> dict[str, object]:
    config = config or TrainConfig()
    set_seed(config.seed)
    device = resolve_device(device_name)
    train_ds, val_ds, test_ds = make_splits(
        train_size=config.train_size,
        val_size=config.val_size,
        test_size=config.test_size,
        seed=config.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)

    model = ToyCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, float | int]] = []
    best_val_acc = -1.0
    best_state = None
    stopped_reason = "max_epochs"
    global_step = 0
    should_stop = False

    def record_metrics(epoch: int, step: int) -> dict[str, float | int]:
        nonlocal best_val_acc, best_state, stopped_reason
        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "step": step,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
        }
        history.append(row)
        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if val_metrics["acc"] >= config.early_stop_acc:
            stopped_reason = f"val_acc >= {config.early_stop_acc}"
        return row

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_loader:
            global_step += 1
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"])
            loss = F.cross_entropy(logits, batch["label"])
            loss.backward()
            optimizer.step()

            if global_step % config.eval_every_steps == 0:
                row = record_metrics(epoch, global_step)
                if float(row["val_acc"]) >= config.early_stop_acc:
                    should_stop = True
                    break
                model.train()
        if should_stop:
            break
        if not history or int(history[-1]["step"]) != global_step:
            row = record_metrics(epoch, global_step)
            if float(row["val_acc"]) >= config.early_stop_acc:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.cpu().state_dict(),
        "config": asdict(config),
        "history": history,
        "test_metrics": test_metrics,
        "stopped_reason": stopped_reason,
        "trainable_parameters": count_trainable_parameters(model),
    }
    torch.save(payload, output)
    metrics_path = output.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps({k: v for k, v in payload.items() if k != "model_state"}, indent=2))
    return {
        "checkpoint": str(output),
        "metrics": str(metrics_path),
        "history": history,
        "test_metrics": test_metrics,
        "stopped_reason": stopped_reason,
        "trainable_parameters": count_trainable_parameters(model),
    }


def load_model(checkpoint: Path, device: torch.device) -> ToyCNN:
    payload = torch.load(checkpoint, map_location=device)
    model = ToyCNN().to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model

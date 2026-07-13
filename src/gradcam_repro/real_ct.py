from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .attribution import METHODS
from .evaluate import score_attributions
from .model import RealCtCNN, ToyCNN, count_trainable_parameters
from .train import resolve_device, set_seed
from .visualize import DEFAULT_METHODS, overlay_heatmap, render_method_grid, sample_z, volume_slice


DEFAULT_RAW_ROOT = Path("data/msd/Task06_Lung")
DEFAULT_CACHE = Path("artifacts/real_ct/msd_lung_presence_48.pt")
DEFAULT_CHECKPOINT = Path("artifacts/real_ct/real_ct_presence_cnn.pt")
DEFAULT_FIGURE_DIR = Path("artifacts/real_ct/figures")
DEFAULT_SCORE_PATH = Path("artifacts/real_ct/real_ct_scores.json")
REAL_CT_FAST_METHODS = ["gradcam", "layercam", "integrated_gradcam"]


@dataclass(frozen=True)
class RealCtPreprocessConfig:
    target_shape: tuple[int, int, int] = (48, 48, 48)
    hu_min: float = -1000.0
    hu_max: float = 400.0
    seed: int = 13
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    task: str = "presence"
    positive_patches_per_case: int = 2
    negative_patches_per_case: int = 2


@dataclass(frozen=True)
class RealCtTrainConfig:
    batch_size: int = 2
    epochs: int = 30
    eval_every_steps: int = 8
    early_stop_acc: float = 0.85
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 13


class RealCtDataset(Dataset[dict[str, torch.Tensor | str]]):
    def __init__(
        self,
        cache: Path,
        split: str = "train",
        limit: int | None = None,
        positive_only: bool = False,
    ) -> None:
        payload = torch.load(cache, map_location="cpu")
        if split not in payload["splits"]:
            raise ValueError(f"Unknown split {split!r}; expected one of {sorted(payload['splits'])}")
        indices = payload["splits"][split]
        if positive_only:
            indices = [idx for idx in indices if int(payload["labels"][idx].item()) == 1]
        if limit is not None:
            indices = indices[:limit]
        self.images = payload["images"][indices]
        self.masks = payload["masks"][indices]
        self.labels = payload["labels"][indices]
        self.centers = payload["centers"][indices]
        self.case_ids = [payload["case_ids"][idx] for idx in indices]

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        return {
            "image": self.images[idx],
            "mask": self.masks[idx],
            "label": self.labels[idx],
            "center": self.centers[idx],
            "case_id": self.case_ids[idx],
        }


def real_batch_to_device(
    batch: dict[str, torch.Tensor | list[str]],
    device: torch.device,
) -> dict[str, torch.Tensor | list[str]]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def evaluate_real_ct(model: ToyCNN, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = real_batch_to_device(batch, device)
            logits = model(batch["image"])  # type: ignore[arg-type]
            labels = batch["label"]  # type: ignore[assignment]
            loss = F.cross_entropy(logits, labels)  # type: ignore[arg-type]
            total_loss += float(loss.item()) * labels.shape[0]  # type: ignore[union-attr]
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())  # type: ignore[arg-type]
            total += int(labels.shape[0])  # type: ignore[union-attr]
    return {"loss": total_loss / max(total, 1), "acc": total_correct / max(total, 1)}


def import_nibabel() -> Any:
    try:
        import nibabel as nib  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "MSD Lung preprocessing requires nibabel. Install it with `uv add nibabel`."
        ) from exc
    return nib


def find_msd_lung_pairs(raw_root: Path) -> list[tuple[str, Path, Path]]:
    images_dir = raw_root / "imagesTr"
    labels_dir = raw_root / "labelsTr"
    if not images_dir.exists() or not labels_dir.exists():
        raise FileNotFoundError(
            f"Expected MSD Task06_Lung with imagesTr/ and labelsTr/ under {raw_root}"
        )
    pairs: list[tuple[str, Path, Path]] = []
    for image_path in sorted(images_dir.glob("*.nii.gz")):
        label_path = labels_dir / image_path.name
        if label_path.exists():
            pairs.append((image_path.name.removesuffix(".nii.gz"), image_path, label_path))
    if not pairs:
        raise FileNotFoundError(f"No matched image/mask pairs found under {raw_root}")
    return pairs


def load_nifti_dhw(path: Path) -> np.ndarray:
    nib = import_nibabel()
    volume_xyz = np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))
    return np.transpose(volume_xyz, (2, 1, 0)).copy()


def normalize_hu(volume: np.ndarray, hu_min: float, hu_max: float) -> np.ndarray:
    clipped = np.clip(volume, hu_min, hu_max)
    return ((clipped - hu_min) / (hu_max - hu_min)).astype(np.float32)


def resize_volume(
    volume: np.ndarray,
    target_shape: tuple[int, int, int],
    mode: str,
) -> torch.Tensor:
    tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    if mode == "nearest":
        resized = F.interpolate(tensor, size=target_shape, mode=mode)
    else:
        resized = F.interpolate(tensor, size=target_shape, mode=mode, align_corners=False)
    return resized.squeeze(0)


def tumor_center(mask: torch.Tensor) -> torch.Tensor | None:
    coords = (mask.squeeze(0) > 0).nonzero()
    if coords.numel() == 0:
        return None
    return coords.float().mean(dim=0).round().to(dtype=torch.long)


def make_left_right_label(center: torch.Tensor, width: int) -> torch.Tensor:
    return torch.tensor(int(center[2].item() >= width / 2), dtype=torch.long)


def crop_or_pad_3d(
    volume: np.ndarray,
    center: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> np.ndarray:
    output = np.zeros(shape, dtype=volume.dtype)
    source_slices = []
    target_slices = []
    for axis, (axis_center, axis_size, axis_target) in enumerate(
        zip(center, volume.shape, shape, strict=True)
    ):
        del axis
        start = int(axis_center) - axis_target // 2
        end = start + axis_target
        source_start = max(0, start)
        source_end = min(axis_size, end)
        target_start = source_start - start
        target_end = target_start + (source_end - source_start)
        source_slices.append(slice(source_start, source_end))
        target_slices.append(slice(target_start, target_end))
    output[tuple(target_slices)] = volume[tuple(source_slices)]
    return output


def choose_negative_center(
    mask: np.ndarray,
    rng: np.random.Generator,
    patch_shape: tuple[int, int, int],
    max_attempts: int = 200,
) -> tuple[int, int, int]:
    depth, height, width = mask.shape
    for _attempt in range(max_attempts):
        center = (
            int(rng.integers(patch_shape[0] // 2, max(patch_shape[0] // 2 + 1, depth - patch_shape[0] // 2))),
            int(rng.integers(patch_shape[1] // 2, max(patch_shape[1] // 2 + 1, height - patch_shape[1] // 2))),
            int(rng.integers(patch_shape[2] // 2, max(patch_shape[2] // 2 + 1, width - patch_shape[2] // 2))),
        )
        mask_patch = crop_or_pad_3d(mask, center, patch_shape)
        if float(mask_patch.sum()) == 0.0:
            return center
    coords = (mask <= 0).nonzero()
    if len(coords[0]) == 0:
        return (depth // 2, height // 2, width // 2)
    idx = int(rng.integers(0, len(coords[0])))
    return (int(coords[0][idx]), int(coords[1][idx]), int(coords[2][idx]))


def make_splits(n_items: int, config: RealCtPreprocessConfig) -> dict[str, list[int]]:
    rng = random.Random(config.seed)
    indices = list(range(n_items))
    rng.shuffle(indices)
    test_count = max(1, round(n_items * config.test_fraction))
    val_count = max(1, round(n_items * config.val_fraction))
    test = indices[:test_count]
    val = indices[test_count : test_count + val_count]
    train = indices[test_count + val_count :]
    return {"train": train, "val": val, "test": test}


def preprocess_msd_lung(
    raw_root: Path,
    output: Path,
    config: RealCtPreprocessConfig | None = None,
) -> dict[str, object]:
    config = config or RealCtPreprocessConfig()
    if config.task == "presence":
        return preprocess_msd_lung_presence(raw_root, output, config)
    pairs = find_msd_lung_pairs(raw_root)
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    centers: list[torch.Tensor] = []
    case_ids: list[str] = []
    source_images: list[str] = []
    source_masks: list[str] = []

    for case_id, image_path, mask_path in pairs:
        image_dhw = normalize_hu(load_nifti_dhw(image_path), config.hu_min, config.hu_max)
        mask_dhw = (load_nifti_dhw(mask_path) > 0).astype(np.float32)
        image_resized = resize_volume(image_dhw, config.target_shape, mode="trilinear")
        mask_resized = (resize_volume(mask_dhw, config.target_shape, mode="nearest") > 0.5).float()
        center = tumor_center(mask_resized)
        if center is None:
            continue
        label = make_left_right_label(center, width=config.target_shape[2])
        images.append(image_resized.contiguous())
        masks.append(mask_resized.contiguous())
        labels.append(label)
        centers.append(center)
        case_ids.append(case_id)
        source_images.append(str(image_path))
        source_masks.append(str(mask_path))

    if not images:
        raise RuntimeError("All MSD Lung cases had empty masks after preprocessing")

    payload = {
        "schema": "gradcam-repro.real-ct-msd-lung.v1",
        "source": {
            "dataset": "MSD Task06_Lung",
            "raw_root": str(raw_root),
            "classification_label": "0=tumor centroid in left half, 1=tumor centroid in right half",
            "segmentation_mask": "MSD lung tumour mask, resized with nearest interpolation",
        },
        "config": asdict(config),
        "images": torch.stack(images).to(dtype=torch.float32),
        "masks": torch.stack(masks).to(dtype=torch.float32),
        "labels": torch.stack(labels).to(dtype=torch.long),
        "centers": torch.stack(centers).to(dtype=torch.long),
        "case_ids": case_ids,
        "source_images": source_images,
        "source_masks": source_masks,
        "splits": make_splits(len(images), config),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    summary = summarize_cache_payload(payload)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    return {"cache": str(output), "summary": str(summary_path), **summary}


def preprocess_msd_lung_presence(
    raw_root: Path,
    output: Path,
    config: RealCtPreprocessConfig,
) -> dict[str, object]:
    pairs = find_msd_lung_pairs(raw_root)
    rng = np.random.default_rng(config.seed)
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    centers: list[torch.Tensor] = []
    case_ids: list[str] = []
    source_images: list[str] = []
    source_masks: list[str] = []

    for case_id, image_path, mask_path in pairs:
        image_dhw = normalize_hu(load_nifti_dhw(image_path), config.hu_min, config.hu_max)
        mask_dhw = (load_nifti_dhw(mask_path) > 0).astype(np.float32)
        coords = np.argwhere(mask_dhw > 0)
        if coords.size == 0:
            continue
        tumor_center_zyx = coords.mean(axis=0)

        for patch_idx in range(config.positive_patches_per_case):
            jitter = rng.integers(
                low=-max(1, config.target_shape[0] // 8),
                high=max(2, config.target_shape[0] // 8 + 1),
                size=3,
            )
            center = tuple((tumor_center_zyx + jitter).round().astype(int).tolist())
            image_patch = crop_or_pad_3d(image_dhw, center, config.target_shape)
            mask_patch = crop_or_pad_3d(mask_dhw, center, config.target_shape)
            if float(mask_patch.sum()) == 0.0:
                center = tuple(tumor_center_zyx.round().astype(int).tolist())
                image_patch = crop_or_pad_3d(image_dhw, center, config.target_shape)
                mask_patch = crop_or_pad_3d(mask_dhw, center, config.target_shape)
            patch_center = tumor_center(torch.from_numpy(mask_patch).unsqueeze(0).float())
            if patch_center is None:
                continue
            images.append(torch.from_numpy(image_patch).unsqueeze(0).float().contiguous())
            masks.append(torch.from_numpy((mask_patch > 0).astype(np.float32)).unsqueeze(0).contiguous())
            labels.append(torch.tensor(1, dtype=torch.long))
            centers.append(patch_center)
            case_ids.append(f"{case_id}_pos{patch_idx}")
            source_images.append(str(image_path))
            source_masks.append(str(mask_path))

        for patch_idx in range(config.negative_patches_per_case):
            center = choose_negative_center(mask_dhw, rng, config.target_shape)
            image_patch = crop_or_pad_3d(image_dhw, center, config.target_shape)
            mask_patch = np.zeros(config.target_shape, dtype=np.float32)
            images.append(torch.from_numpy(image_patch).unsqueeze(0).float().contiguous())
            masks.append(torch.from_numpy(mask_patch).unsqueeze(0).contiguous())
            labels.append(torch.tensor(0, dtype=torch.long))
            centers.append(torch.tensor([dim // 2 for dim in config.target_shape], dtype=torch.long))
            case_ids.append(f"{case_id}_neg{patch_idx}")
            source_images.append(str(image_path))
            source_masks.append(str(mask_path))

    if not images:
        raise RuntimeError("No tumour-presence patches were created from MSD Lung")

    payload = {
        "schema": "gradcam-repro.real-ct-msd-lung.v2",
        "source": {
            "dataset": "MSD Task06_Lung",
            "raw_root": str(raw_root),
            "classification_label": "0=non-tumour CT patch, 1=tumour-present CT patch",
            "segmentation_mask": "MSD lung tumour mask cropped into positive patches",
            "task": "tumour_presence_patch",
        },
        "config": asdict(config),
        "images": torch.stack(images).to(dtype=torch.float32),
        "masks": torch.stack(masks).to(dtype=torch.float32),
        "labels": torch.stack(labels).to(dtype=torch.long),
        "centers": torch.stack(centers).to(dtype=torch.long),
        "case_ids": case_ids,
        "source_images": source_images,
        "source_masks": source_masks,
        "splits": make_splits(len(images), config),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    summary = summarize_cache_payload(payload)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    return {"cache": str(output), "summary": str(summary_path), **summary}


def summarize_cache_payload(payload: dict[str, Any]) -> dict[str, object]:
    labels = payload["labels"]
    label_counts = {
        str(label): int((labels == label).sum().item())
        for label in sorted(set(labels.detach().cpu().tolist()))
    }
    return {
        "schema": payload["schema"],
        "n_cases": int(labels.shape[0]),
        "image_shape": list(payload["images"].shape[1:]),
        "label_counts": label_counts,
        "splits": {name: len(indices) for name, indices in payload["splits"].items()},
        "source": payload["source"],
    }


def load_cache_summary(cache: Path) -> dict[str, object]:
    payload = torch.load(cache, map_location="cpu")
    return summarize_cache_payload(payload)


def train_real_ct_model(
    cache: Path,
    output: Path,
    config: RealCtTrainConfig | None = None,
    device_name: str = "auto",
) -> dict[str, object]:
    config = config or RealCtTrainConfig()
    set_seed(config.seed)
    device = resolve_device(device_name)
    train_ds = RealCtDataset(cache, split="train")
    val_ds = RealCtDataset(cache, split="val")
    test_ds = RealCtDataset(cache, split="test")
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)
    model = RealCtCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history: list[dict[str, float | int]] = []
    best_state = None
    best_val_acc = -1.0
    stopped_reason = "max_epochs"
    global_step = 0
    should_stop = False

    def record_metrics(epoch: int, step: int) -> dict[str, float | int]:
        nonlocal best_state, best_val_acc, stopped_reason
        train_metrics = evaluate_real_ct(model, train_loader, device)
        val_metrics = evaluate_real_ct(model, val_loader, device)
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
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        if val_metrics["acc"] >= config.early_stop_acc:
            stopped_reason = f"val_acc >= {config.early_stop_acc}"
        return row

    for epoch in range(1, config.epochs + 1):
        model.train()
        for batch in train_loader:
            global_step += 1
            batch = real_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["image"])  # type: ignore[arg-type]
            loss = F.cross_entropy(logits, batch["label"])  # type: ignore[arg-type]
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
    test_metrics = evaluate_real_ct(model, test_loader, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.cpu().state_dict(),
        "config": asdict(config),
        "history": history,
        "test_metrics": test_metrics,
        "stopped_reason": stopped_reason,
        "trainable_parameters": count_trainable_parameters(model),
        "data_cache": str(cache),
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


def load_real_ct_model(checkpoint: Path, device: torch.device) -> RealCtCNN:
    payload = torch.load(checkpoint, map_location=device)
    model = RealCtCNN().to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def render_real_ct_grid(
    model: ToyCNN,
    dataset: RealCtDataset,
    output: Path,
    device: torch.device,
    sample_count: int = 3,
    methods: list[str] | None = None,
) -> None:
    methods = methods or REAL_CT_FAST_METHODS
    render_method_grid(
        model=model,
        dataset=dataset,  # type: ignore[arg-type]
        output=output,
        device=device,
        sample_count=min(sample_count, len(dataset)),
        methods=methods,
    )


def score_real_ct_attributions(
    model: ToyCNN,
    dataset: RealCtDataset,
    device: torch.device,
    methods: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    return score_attributions(
        model=model,
        dataset=dataset,  # type: ignore[arg-type]
        device=device,
        methods=methods or REAL_CT_FAST_METHODS,
    )


def render_real_ct_single_case(
    model: ToyCNN,
    dataset: RealCtDataset,
    output: Path,
    device: torch.device,
    sample_idx: int = 0,
    methods: list[str] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    methods = methods or REAL_CT_FAST_METHODS
    sample = dataset[sample_idx]
    z = sample_z(sample)  # type: ignore[arg-type]
    image = sample["image"].unsqueeze(0).to(device)  # type: ignore[union-attr]
    label = sample["label"].view(1).to(device)  # type: ignore[union-attr]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(methods) + 2, figsize=(2.6 * (len(methods) + 2), 2.7))
    axes[0].imshow(volume_slice(sample["image"], z), cmap="gray", vmin=0, vmax=1)  # type: ignore[arg-type]
    axes[0].set_title(f"{sample['case_id']}\ny={int(sample['label'])}", fontsize=9)
    axes[1].imshow(volume_slice(sample["mask"], z), cmap="Greens", vmin=0, vmax=1)  # type: ignore[arg-type]
    axes[1].set_title("tumor mask", fontsize=9)
    for axis, method in zip(axes[2:], methods, strict=True):
        heatmap = METHODS[method](model, image, label)
        axis.imshow(overlay_heatmap(sample["image"], heatmap.cpu(), z))  # type: ignore[arg-type]
        axis.set_title(method.replace("_", "\n"), fontsize=9)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.tight_layout(pad=0.3)
    fig.savefig(output, dpi=180)
    plt.close(fig)

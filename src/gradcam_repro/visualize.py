from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "gradcam-repro-matplotlib"))

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch

from .attribution import METHODS, guided_backprop_raw, normalize_map, score_for_target, upsample_to_input
from .data import CrossHalfDataset
from .model import ToyCNN


DEFAULT_METHODS = [
    "notgradcam",
    "gradcam",
    "guided_gradcam",
    "layercam",
    "occlusion",
    "integrated_gradients",
    "integrated_gradcam",
]


def superscript_number(value: int) -> str:
    table = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
    return str(value).translate(table)


def sample_z(sample: dict[str, torch.Tensor]) -> int:
    return int(sample["center"][0].item())


def feature_z(sample: dict[str, torch.Tensor], feature: torch.Tensor) -> int:
    input_depth = int(sample["image"].shape[-3])
    feature_depth = int(feature.shape[-3])
    if input_depth <= 1:
        return 0
    z = sample_z(sample)
    return max(0, min(feature_depth - 1, round(z * (feature_depth - 1) / (input_depth - 1))))


def volume_slice(tensor: torch.Tensor, z: int | None = None) -> torch.Tensor:
    data = tensor.detach().cpu().squeeze()
    if data.ndim == 3:
        z = data.shape[0] // 2 if z is None else max(0, min(data.shape[0] - 1, int(z)))
        return data[z]
    return data


def overlay_heatmap(image: torch.Tensor, heatmap: torch.Tensor, z: int | None = None) -> torch.Tensor:
    image_np = volume_slice(image, z).numpy()
    heat_np = volume_slice(heatmap, z).numpy()
    cmap = plt.get_cmap("turbo")
    color = cmap(heat_np)[..., :3]
    gray = image_np[..., None].repeat(3, axis=2)
    overlay = 0.55 * gray + 0.45 * color
    return torch.from_numpy(overlay.clip(0, 1))


def normalize_panel(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    data = tensor.detach().float().cpu().squeeze()
    min_value = data.min()
    max_value = data.max()
    return (data - min_value) / (max_value - min_value + eps)


def _cam_forward_for_viz(model: ToyCNN, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits, features = model(x, return_features=True)
    activation = features["stage2"]
    activation.retain_grad()
    return logits, activation


def _sample_for_viz(
    dataset: CrossHalfDataset,
    sample_idx: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    sample = dataset[sample_idx]
    image = sample["image"].unsqueeze(0).to(device)
    target = sample["label"].view(1).to(device)
    return sample, image, target


def render_panel_figure(
    output: Path,
    title: str,
    panels: list[tuple[str, torch.Tensor, str | None, float | None, float | None]],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(14.8, 4.3), squeeze=False)
    flat_axes = axes.flatten()
    for axis, panel in zip(flat_axes, panels, strict=False):
        panel_title, data, cmap, vmin, vmax = panel
        if cmap is None:
            axis.imshow(data.detach().cpu())
        else:
            axis.imshow(data.detach().cpu().squeeze(), cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(panel_title, fontsize=10)
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in flat_axes[len(panels) :]:
        axis.axis("off")
    for idx in range(min(len(panels) - 1, 4)):
        flat_axes[idx].text(
            1.08,
            0.5,
            "→",
            transform=flat_axes[idx].transAxes,
            fontsize=17,
            color="#5E6873",
            va="center",
            ha="center",
        )
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(pad=0.35)
    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def render_method_grid(
    model: ToyCNN,
    dataset: CrossHalfDataset,
    output: Path,
    device: torch.device,
    sample_count: int = 3,
    methods: list[str] | None = None,
) -> None:
    methods = methods or DEFAULT_METHODS
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        sample_count,
        len(methods) + 2,
        figsize=(2.45 * (len(methods) + 2), 2.45 * sample_count),
        squeeze=False,
    )
    for row_idx in range(sample_count):
        sample = dataset[row_idx]
        z = sample_z(sample)
        image = sample["image"].unsqueeze(0).to(device)
        label = sample["label"].view(1).to(device)
        gt = sample["mask"]
        axes[row_idx, 0].imshow(volume_slice(sample["image"], z), cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 0].set_title(f"3D input y={int(sample['label'])}\nz={z}", fontsize=11)
        axes[row_idx, 1].imshow(volume_slice(gt, z), cmap="Greens", vmin=0, vmax=1)
        axes[row_idx, 1].set_title("gt volume\nslice", fontsize=11)
        for col_idx, name in enumerate(methods, start=2):
            heatmap = METHODS[name](model, image, label)
            overlay = overlay_heatmap(sample["image"], heatmap.cpu(), z)
            axes[row_idx, col_idx].imshow(overlay)
            axes[row_idx, col_idx].set_title(name.replace("_", "\n"), fontsize=11)
        for axis in axes[row_idx]:
            axis.set_xticks([])
            axis.set_yticks([])
    fig.tight_layout(pad=0.35)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def render_class_discriminability(
    model: ToyCNN,
    dataset: CrossHalfDataset,
    output: Path,
    device: torch.device,
    sample_idx: int = 0,
    methods: list[str] | None = None,
) -> None:
    methods = methods or ["notgradcam", "gradcam", "layercam", "integrated_gradcam"]
    sample = dataset[sample_idx]
    z = sample_z(sample)
    image = sample["image"].unsqueeze(0).to(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        2,
        len(methods) + 2,
        figsize=(2.1 * (len(methods) + 2), 4.2),
        squeeze=False,
    )
    for row_idx, target in enumerate([0, 1]):
        axes[row_idx, 0].imshow(volume_slice(sample["image"], z), cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 0].set_title(f"3D input true={int(sample['label'])}\nz={z}", fontsize=8)
        axes[row_idx, 1].imshow(volume_slice(sample["mask"], z), cmap="Greens", vmin=0, vmax=1)
        axes[row_idx, 1].set_title("gt volume slice", fontsize=8)
        for col_idx, name in enumerate(methods, start=2):
            heatmap = METHODS[name](model, image, target)
            overlay = overlay_heatmap(sample["image"], heatmap.cpu(), z)
            title_color = "green" if target == int(sample["label"]) else "red"
            axes[row_idx, col_idx].imshow(overlay)
            axes[row_idx, col_idx].set_title(f"{name}\nclass {target}", fontsize=8, color=title_color)
        for axis in axes[row_idx]:
            axis.set_xticks([])
            axis.set_yticks([])
    fig.tight_layout(pad=0.4)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def render_notgradcam_decomposition(
    model: ToyCNN,
    dataset: CrossHalfDataset,
    output: Path,
    device: torch.device,
    sample_idx: int = 0,
    channel_count: int = 4,
) -> None:
    sample = dataset[sample_idx]
    z = sample_z(sample)
    image = sample["image"].unsqueeze(0).to(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        _, features = model(image, return_features=True)
        activation = features["stage2"]
        channel_scores = activation.flatten(2).mean(dim=2).squeeze(0)
        top_channels = torch.topk(channel_scores, k=min(channel_count, activation.shape[1])).indices
        sum_map = activation.sum(dim=1, keepdim=True)
        mean_map = activation.mean(dim=1, keepdim=True)
        output_map = normalize_map(upsample_to_input(mean_map, image))
        fz = feature_z(sample, activation)

    overlay = overlay_heatmap(sample["image"], output_map.cpu(), z)
    top_row = [
        (f"input x[z={z}]", volume_slice(sample["image"], z), "gray", 0, 1),
        *[
            (
                f"A{superscript_number(int(channel))}[z={fz}]",
                volume_slice(activation[0, int(channel)], fz),
                "viridis",
                None,
                None,
            )
            for channel in top_channels
        ],
    ]
    bottom_row = [
        ("Σₖ Aᵏ", volume_slice(sum_map, fz), "viridis", None, None),
        ("(1/K)Σₖ Aᵏ", volume_slice(mean_map, fz), "viridis", None, None),
        ("Up(mean)", volume_slice(output_map, z), "turbo", 0, 1),
        ("activation output", overlay, None, None, None),
        ("target volume slice", volume_slice(sample["mask"], z), "Greens", 0, 1),
    ]
    panels = [top_row, bottom_row]
    fig, axes = plt.subplots(2, 5, figsize=(14.8, 4.3), squeeze=False)
    for row_axes, row_panels in zip(axes, panels, strict=True):
        for axis, (title, data, cmap, vmin, vmax) in zip(row_axes, row_panels, strict=True):
            if cmap is None:
                axis.imshow(data)
            else:
                axis.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
            axis.set_title(title, fontsize=10)
            axis.set_xticks([])
            axis.set_yticks([])
    for axis in axes.flatten()[len(top_row) + len(bottom_row) :]:
        axis.axis("off")
    for axis in axes[1, :3]:
        axis.text(
            1.08,
            0.5,
            "→",
            transform=axis.transAxes,
            fontsize=17,
            color="#5E6873",
            va="center",
            ha="center",
        )
    fig.suptitle("3D notGradCAM: M(x) = Up( (1/K) Σₖ Aᵏ(x) ), axial slice shown", fontsize=13, fontweight="bold")
    fig.tight_layout(pad=0.35)
    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def render_gradcam_decomposition(
    model: ToyCNN,
    dataset: CrossHalfDataset,
    output: Path,
    device: torch.device,
    sample_idx: int = 0,
) -> None:
    sample, image, target = _sample_for_viz(dataset, sample_idx, device)
    z = sample_z(sample)
    model.eval()
    model.zero_grad(set_to_none=True)
    logits, activation = _cam_forward_for_viz(model, image)
    score_for_target(logits, target).backward()
    gradients = activation.grad
    if gradients is None:
        raise RuntimeError("Grad-CAM detail figure did not receive gradients")
    weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
    channel = int(weights.abs().flatten().argmax())
    weighted = weights * activation
    cam = weighted.sum(dim=1, keepdim=True)
    heatmap = normalize_map(upsample_to_input(cam.clamp_min(0), image))
    fz = feature_z(sample, activation)
    panels = [
        (f"input x[z={z}]", volume_slice(sample["image"], z), "gray", 0, 1),
        (f"A{superscript_number(channel)}[z={fz}]", normalize_panel(volume_slice(activation[0, channel], fz)), "viridis", 0, 1),
        (
            f"∂yᶜ/∂A{superscript_number(channel)}",
            normalize_panel(volume_slice(gradients[0, channel], fz)),
            "coolwarm",
            0,
            1,
        ),
        (
            f"α{superscript_number(channel)}A{superscript_number(channel)}",
            normalize_panel(volume_slice(weighted[0, channel], fz)),
            "viridis",
            0,
            1,
        ),
        ("∑ₖ αₖᶜAᵏ", normalize_panel(volume_slice(cam, fz)), "viridis", 0, 1),
        ("ReLU + Up", volume_slice(heatmap, z), "turbo", 0, 1),
        ("Grad-CAM output", overlay_heatmap(sample["image"], heatmap.cpu(), z), None, None, None),
        ("target volume slice", volume_slice(sample["mask"], z), "Greens", 0, 1),
    ]
    render_panel_figure(output, "3D Grad-CAM: gradients weight feature volumes; axial slice shown", panels)


def render_guided_gradcam_decomposition(
    model: ToyCNN,
    dataset: CrossHalfDataset,
    output: Path,
    device: torch.device,
    sample_idx: int = 0,
) -> None:
    sample, image, target = _sample_for_viz(dataset, sample_idx, device)
    z = sample_z(sample)
    model.eval()
    guided_raw = guided_backprop_raw(model, image, target)
    guided = normalize_map(guided_raw.abs().sum(dim=1, keepdim=True))
    cam = METHODS["gradcam"](model, image, target)
    product = normalize_map((guided_raw * cam).abs().sum(dim=1, keepdim=True))
    panels = [
        (f"input x[z={z}]", volume_slice(sample["image"], z), "gray", 0, 1),
        ("GuidedBP(x,c)", volume_slice(guided, z), "magma", 0, 1),
        ("Grad-CAM Gᶜ", volume_slice(cam, z), "turbo", 0, 1),
        ("GuidedBP ⊙ Gᶜ", volume_slice(product, z), "turbo", 0, 1),
        ("guided output", overlay_heatmap(sample["image"], product.cpu(), z), None, None, None),
        ("target volume slice", volume_slice(sample["mask"], z), "Greens", 0, 1),
    ]
    render_panel_figure(output, "3D Guided Grad-CAM: GuidedBP volume masked by 3D Grad-CAM", panels)


def render_layercam_decomposition(
    model: ToyCNN,
    dataset: CrossHalfDataset,
    output: Path,
    device: torch.device,
    sample_idx: int = 0,
) -> None:
    sample, image, target = _sample_for_viz(dataset, sample_idx, device)
    z = sample_z(sample)
    model.eval()
    model.zero_grad(set_to_none=True)
    logits, activation = _cam_forward_for_viz(model, image)
    score_for_target(logits, target).backward()
    gradients = activation.grad
    if gradients is None:
        raise RuntimeError("LayerCAM detail figure did not receive gradients")
    positive_grad = gradients.clamp_min(0)
    local = positive_grad * activation
    channel = int(local.flatten(2).mean(dim=2).squeeze(0).argmax())
    summed = local.sum(dim=1, keepdim=True).clamp_min(0)
    heatmap = normalize_map(upsample_to_input(summed, image))
    fz = feature_z(sample, activation)
    signed_grad_slice = volume_slice(gradients[0, channel], fz)
    signed_limit = float(signed_grad_slice.detach().abs().max().clamp_min(1e-8).item())
    panels = [
        (f"input x[z={z}]", volume_slice(sample["image"], z), "gray", 0, 1),
        (f"A{superscript_number(channel)}[z={fz}]", normalize_panel(volume_slice(activation[0, channel], fz)), "viridis", 0, 1),
        (
            f"δyᶜ/δA{superscript_number(channel)}",
            signed_grad_slice,
            "coolwarm",
            -signed_limit,
            signed_limit,
        ),
        (
            f"ReLU(δyᶜ/δA{superscript_number(channel)})",
            normalize_panel(volume_slice(positive_grad[0, channel], fz)),
            "magma",
            0,
            1,
        ),
        ("ReLU(δyᶜ/δAᵏ) ⊙ Aᵏ", normalize_panel(volume_slice(local[0, channel], fz)), "viridis", 0, 1),
        ("∑ₖ local evidence", normalize_panel(volume_slice(summed, fz)), "viridis", 0, 1),
        ("LayerCAM output", overlay_heatmap(sample["image"], heatmap.cpu(), z), None, None, None),
        ("target volume slice", volume_slice(sample["mask"], z), "Greens", 0, 1),
    ]
    render_panel_figure(output, "3D LayerCAM: backprop δy, keep positive local gradients, then sum evidence", panels)


def render_occlusion_decomposition(
    model: ToyCNN,
    dataset: CrossHalfDataset,
    output: Path,
    device: torch.device,
    sample_idx: int = 0,
    mask_size: int = 4,
    stride: int = 2,
) -> None:
    sample, image, target = _sample_for_viz(dataset, sample_idx, device)
    z = sample_z(sample)
    model.eval()
    mask = sample["mask"].squeeze()
    coords = mask.nonzero()
    center_z = int(coords[:, 0].float().mean().round().item())
    center_y = int(coords[:, 1].float().mean().round().item())
    center_x = int(coords[:, 2].float().mean().round().item())
    depth, height, width = image.shape[-3:]
    positions = [
        (
            max(0, min(depth - mask_size, center_z - mask_size // 2)),
            max(0, min(height - mask_size, center_y - mask_size // 2)),
            max(0, min(width - mask_size, center_x - mask_size // 2)),
        ),
        (z, 4, 4),
        (z, height - mask_size - 4, width - mask_size - 4),
    ]
    occluded_panels = []
    fill_value = float(image.mean().item())
    for idx, (z0, y, x0) in enumerate(positions, start=1):
        occluded = image.detach().clone()
        occluded[:, :, z0 : z0 + mask_size, y : y + mask_size, x0 : x0 + mask_size] = fill_value
        occluded_panels.append((f"mean-fill p{idx}", volume_slice(occluded, z), "gray", 0, 1))

    grid_slice = volume_slice(sample["image"], z).detach().float().cpu()
    grid_rgb = grid_slice.unsqueeze(-1).repeat(1, 1, 3)
    dot_color = torch.tensor([1.0, 0.32, 0.0])
    z_starts = range(0, depth - mask_size + 1, stride)
    y_starts = range(0, height - mask_size + 1, stride)
    x_starts = range(0, width - mask_size + 1, stride)
    for z0 in z_starts:
        if not (z0 <= z < z0 + mask_size):
            continue
        for y0 in y_starts:
            cy = min(height - 1, y0 + mask_size // 2)
            for x0 in x_starts:
                cx = min(width - 1, x0 + mask_size // 2)
                grid_rgb[cy, cx] = dot_color

    heatmap = METHODS["occlusion"](model, image, target, mask_size=mask_size, stride=stride)
    panels = [
        (f"input x[z={z}]", volume_slice(sample["image"], z), "gray", 0, 1),
        (f"full sweep grid\ns={stride}", grid_rgb, None, None, None),
        *occluded_panels[:2],
        ("positive H(p)\nscore drop", volume_slice(heatmap, z), "turbo", 0, 1),
        ("occlusion output", overlay_heatmap(sample["image"], heatmap.cpu(), z), None, None, None),
        ("target volume slice", volume_slice(sample["mask"], z), "Greens", 0, 1),
    ]
    render_panel_figure(output, "3D Occlusion: full-volume cube sweep, then keep positive score drops", panels)


def render_integrated_gradients_decomposition(
    model: ToyCNN,
    dataset: CrossHalfDataset,
    output: Path,
    device: torch.device,
    sample_idx: int = 0,
    steps: int = 32,
) -> None:
    sample, image, target = _sample_for_viz(dataset, sample_idx, device)
    z = sample_z(sample)
    model.eval()
    baseline = torch.zeros_like(image)
    total_grad = torch.zeros_like(image)
    midpoint = baseline + 0.5 * (image - baseline)
    for alpha in torch.linspace(0, 1, steps + 1, device=image.device, dtype=image.dtype)[1:]:
        model.zero_grad(set_to_none=True)
        x_step = (baseline + alpha * (image - baseline)).detach().requires_grad_(True)
        logits = model(x_step)
        grad = torch.autograd.grad(score_for_target(logits, target), x_step)[0]
        total_grad += grad
    avg_grad = total_grad / steps
    attribution = (image - baseline) * avg_grad
    raw_heatmap = normalize_map(attribution.abs().sum(dim=1, keepdim=True))
    heatmap = METHODS["integrated_gradients"](model, image, target)
    panels = [
        ("baseline x′", volume_slice(baseline, z), "gray", 0, 1),
        (f"input x[z={z}]", volume_slice(sample["image"], z), "gray", 0, 1),
        ("path α=0.5", volume_slice(midpoint, z), "gray", 0, 1),
        ("avg input grad", normalize_panel(volume_slice(avg_grad.abs().sum(dim=1, keepdim=True), z)), "magma", 0, 1),
        ("(x−x′)·avg grad", volume_slice(raw_heatmap, z), "turbo", 0, 1),
        ("IG output", overlay_heatmap(sample["image"], heatmap.cpu(), z), None, None, None),
        ("target volume slice", volume_slice(sample["mask"], z), "Greens", 0, 1),
    ]
    render_panel_figure(output, "3D Integrated Gradients: integrate input-volume gradients along a baseline path", panels)


def render_integrated_gradcam_decomposition(
    model: ToyCNN,
    dataset: CrossHalfDataset,
    output: Path,
    device: torch.device,
    sample_idx: int = 0,
    steps: int = 20,
) -> None:
    sample, image, target = _sample_for_viz(dataset, sample_idx, device)
    z = sample_z(sample)
    model.eval()
    baseline = torch.ones_like(image) * image.detach().mean(dim=(1, 2, 3, 4), keepdim=True)
    with torch.no_grad():
        _, input_features = model(image, return_features=True)
        _, baseline_features = model(baseline, return_features=True)
        feature_delta = input_features["stage2"] - baseline_features["stage2"]

    total_grad = torch.zeros_like(feature_delta)
    for alpha in torch.linspace(0, 1, steps + 1, device=image.device, dtype=image.dtype)[1:]:
        model.zero_grad(set_to_none=True)
        x_step = (baseline + alpha * (image - baseline)).detach()
        logits, activation = _cam_forward_for_viz(model, x_step)
        score_for_target(logits, target).backward()
        gradients = activation.grad
        if gradients is None:
            raise RuntimeError("Integrated Grad-CAM detail figure did not receive gradients")
        total_grad += gradients
    avg_grad = total_grad / steps
    contribution = avg_grad * feature_delta
    summed = contribution.abs().sum(dim=1, keepdim=True)
    heatmap = METHODS["integrated_gradcam"](model, image, target)
    fz = feature_z(sample, feature_delta)
    panels = [
        ("baseline x′", volume_slice(baseline, z), "gray", 0, 1),
        (f"input x[z={z}]", volume_slice(sample["image"], z), "gray", 0, 1),
        ("ΔA mean", normalize_panel(volume_slice(feature_delta.mean(dim=1, keepdim=True), fz)), "viridis", 0, 1),
        ("avg ∂yᶜ/∂A", normalize_panel(volume_slice(avg_grad.mean(dim=1, keepdim=True), fz)), "magma", 0, 1),
        ("ΔA · avg grad", normalize_panel(volume_slice(summed, fz)), "viridis", 0, 1),
        ("IGC output", overlay_heatmap(sample["image"], heatmap.cpu(), z), None, None, None),
        ("target volume slice", volume_slice(sample["mask"], z), "Greens", 0, 1),
    ]
    render_panel_figure(output, "3D Integrated Grad-CAM: path-integrated feature-volume gradients weight ΔA", panels)


DETAIL_RENDERERS = {
    "gradcam": render_gradcam_decomposition,
    "guided_gradcam": render_guided_gradcam_decomposition,
    "layercam": render_layercam_decomposition,
    "occlusion": render_occlusion_decomposition,
    "integrated_gradients": render_integrated_gradients_decomposition,
    "integrated_gradcam": render_integrated_gradcam_decomposition,
}

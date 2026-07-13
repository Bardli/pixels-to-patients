from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from .model import ToyCNN


TARGET_LAYER = "stage2"


def normalize_map(heatmap: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    flat = heatmap.flatten(1)
    view_shape = (heatmap.shape[0],) + (1,) * (heatmap.ndim - 1)
    mins = flat.min(dim=1).values.view(view_shape)
    maxs = flat.max(dim=1).values.view(view_shape)
    return (heatmap - mins) / (maxs - mins + eps)


def upsample_to_input(heatmap: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.interpolate(heatmap, size=x.shape[-3:], mode="trilinear", align_corners=False)


def score_for_target(logits: torch.Tensor, target: int | torch.Tensor | None) -> torch.Tensor:
    if target is None:
        target_tensor = logits.argmax(dim=1)
    elif isinstance(target, int):
        target_tensor = torch.full((logits.shape[0],), target, device=logits.device, dtype=torch.long)
    else:
        target_tensor = target.to(device=logits.device, dtype=torch.long)
    return logits.gather(1, target_tensor.view(-1, 1)).sum()


def _cam_forward(
    model: ToyCNN, x: torch.Tensor, layer: str = TARGET_LAYER
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, features = model(x, return_features=True)
    activation = features[layer]
    activation.retain_grad()
    return logits, activation


def notgradcam(
    model: ToyCNN, x: torch.Tensor, target: int | torch.Tensor | None = None, layer: str = TARGET_LAYER
) -> torch.Tensor:
    del target
    model.eval()
    with torch.no_grad():
        _, features = model(x, return_features=True)
        cam = features[layer].mean(dim=1, keepdim=True)
    return normalize_map(upsample_to_input(cam, x))


def gradcam(
    model: ToyCNN,
    x: torch.Tensor,
    target: int | torch.Tensor | None = None,
    layer: str = TARGET_LAYER,
    normalize: bool = True,
) -> torch.Tensor:
    model.eval()
    model.zero_grad(set_to_none=True)
    logits, activation = _cam_forward(model, x, layer)
    score_for_target(logits, target).backward()
    gradients = activation.grad
    if gradients is None:
        raise RuntimeError("Grad-CAM hook did not receive activation gradients")
    weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True).clamp_min(0)
    cam = upsample_to_input(cam, x)
    return normalize_map(cam) if normalize else cam


def layercam(
    model: ToyCNN,
    x: torch.Tensor,
    target: int | torch.Tensor | None = None,
    layer: str = TARGET_LAYER,
) -> torch.Tensor:
    model.eval()
    model.zero_grad(set_to_none=True)
    logits, activation = _cam_forward(model, x, layer)
    score_for_target(logits, target).backward()
    gradients = activation.grad
    if gradients is None:
        raise RuntimeError("LayerCAM hook did not receive activation gradients")
    cam = (gradients.clamp_min(0) * activation).sum(dim=1, keepdim=True).clamp_min(0)
    return normalize_map(upsample_to_input(cam, x))


@contextmanager
def guided_relu_hooks(model: nn.Module):
    activations: dict[nn.Module, torch.Tensor] = {}
    handles = []

    def forward_hook(module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations[module] = output

    def backward_hook(
        module: nn.Module,
        _grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> tuple[torch.Tensor]:
        grad = grad_output[0]
        if grad is None:
            raise RuntimeError("Guided ReLU hook received no gradient")
        activation = activations[module]
        guided_grad = grad.clamp_min(0) * (activation > 0).to(dtype=grad.dtype)
        return (guided_grad,)

    for module in model.modules():
        if isinstance(module, nn.ReLU):
            handles.append(module.register_forward_hook(forward_hook))
            handles.append(module.register_full_backward_hook(backward_hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def guided_backprop_raw(
    model: ToyCNN, x: torch.Tensor, target: int | torch.Tensor | None = None
) -> torch.Tensor:
    model.eval()
    model.zero_grad(set_to_none=True)
    x_req = x.detach().clone().requires_grad_(True)
    with guided_relu_hooks(model):
        logits = model(x_req)
        score_for_target(logits, target).backward()
    if x_req.grad is None:
        raise RuntimeError("Guided backprop did not produce an input gradient")
    return x_req.grad


def guided_backprop(
    model: ToyCNN, x: torch.Tensor, target: int | torch.Tensor | None = None
) -> torch.Tensor:
    return normalize_map(guided_backprop_raw(model, x, target).abs().sum(dim=1, keepdim=True))


def guided_gradcam(
    model: ToyCNN,
    x: torch.Tensor,
    target: int | torch.Tensor | None = None,
    layer: str = TARGET_LAYER,
) -> torch.Tensor:
    cam = gradcam(model, x, target, layer=layer, normalize=True)
    guided = guided_backprop_raw(model, x, target)
    product = guided * cam
    return normalize_map(product.abs().sum(dim=1, keepdim=True))


def gaussian_blur3d(heatmap: torch.Tensor, sigma: float = 0.8) -> torch.Tensor:
    if sigma <= 0:
        return heatmap
    radius = max(1, int(3 * sigma))
    coords = torch.arange(-radius, radius + 1, device=heatmap.device, dtype=heatmap.dtype)
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_3d = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
    kernel = kernel_3d.view(1, 1, *kernel_3d.shape)
    return F.conv3d(heatmap, kernel, padding=radius)


def integrated_gradients(
    model: ToyCNN,
    x: torch.Tensor,
    target: int | torch.Tensor | None = None,
    steps: int = 50,
    smooth_sigma: float = 0.8,
) -> torch.Tensor:
    model.eval()
    baseline = torch.zeros_like(x)
    total_grad = torch.zeros_like(x)
    for alpha in torch.linspace(0, 1, steps + 1, device=x.device, dtype=x.dtype)[1:]:
        model.zero_grad(set_to_none=True)
        x_step = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        logits = model(x_step)
        grad = torch.autograd.grad(score_for_target(logits, target), x_step)[0]
        total_grad += grad
    attribution = (x - baseline) * total_grad / steps
    heatmap = attribution.abs().sum(dim=1, keepdim=True)
    heatmap = gaussian_blur3d(heatmap, sigma=smooth_sigma)
    return normalize_map(heatmap)


def integrated_gradcam(
    model: ToyCNN,
    x: torch.Tensor,
    target: int | torch.Tensor | None = None,
    layer: str = TARGET_LAYER,
    steps: int = 20,
) -> torch.Tensor:
    model.eval()
    baseline = torch.ones_like(x) * x.detach().mean(dim=(1, 2, 3, 4), keepdim=True)
    with torch.no_grad():
        _, input_features = model(x, return_features=True)
        _, baseline_features = model(baseline, return_features=True)
        feature_delta = input_features[layer] - baseline_features[layer]

    total_grad = torch.zeros_like(feature_delta)
    for alpha in torch.linspace(0, 1, steps + 1, device=x.device, dtype=x.dtype)[1:]:
        model.zero_grad(set_to_none=True)
        x_step = (baseline + alpha * (x - baseline)).detach()
        logits, activation = _cam_forward(model, x_step, layer)
        score_for_target(logits, target).backward()
        gradients = activation.grad
        if gradients is None:
            raise RuntimeError("Integrated Grad-CAM hook did not receive activation gradients")
        total_grad += gradients

    contribution = (total_grad / steps) * feature_delta
    cam = contribution.abs().sum(dim=1, keepdim=True)
    return normalize_map(upsample_to_input(cam, x))


def occlusion_sensitivity(
    model: ToyCNN,
    x: torch.Tensor,
    target: int | torch.Tensor | None = None,
    mask_size: int = 4,
    stride: int = 4,
    fill_value: float | None = None,
    batch_size: int = 64,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        if target is None:
            target_tensor = logits.argmax(dim=1)
        elif isinstance(target, int):
            target_tensor = torch.full((x.shape[0],), target, device=x.device, dtype=torch.long)
        else:
            target_tensor = target.to(device=x.device, dtype=torch.long)
        base_probs = logits.softmax(dim=1).gather(1, target_tensor.view(-1, 1)).view(-1)

    batch, _, depth, height, width = x.shape
    heatmap = torch.zeros((batch, 1, depth, height, width), device=x.device, dtype=x.dtype)
    counts = torch.zeros_like(heatmap)
    fill_tensor = (
        x.detach().mean(dim=(1, 2, 3, 4), keepdim=True)
        if fill_value is None
        else None
    )
    positions = [
        (z, y, x0)
        for z in range(0, depth - mask_size + 1, stride)
        for y in range(0, height - mask_size + 1, stride)
        for x0 in range(0, width - mask_size + 1, stride)
    ]

    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        occluded = x.repeat_interleave(len(chunk), dim=0).clone()
        for sample_idx in range(batch):
            offset = sample_idx * len(chunk)
            for j, (z, y, x0) in enumerate(chunk):
                replacement = fill_value if fill_tensor is None else fill_tensor[sample_idx]
                occluded[
                    offset + j,
                    :,
                    z : z + mask_size,
                    y : y + mask_size,
                    x0 : x0 + mask_size,
                ] = replacement
        with torch.no_grad():
            probs = model(occluded).softmax(dim=1)
        probs = probs.view(batch, len(chunk), -1)
        target_probs = probs.gather(
            2, target_tensor.view(batch, 1, 1).expand(batch, len(chunk), 1)
        ).squeeze(2)
        drops = base_probs[:, None] - target_probs
        for j, (z, y, x0) in enumerate(chunk):
            heatmap[
                :,
                :,
                z : z + mask_size,
                y : y + mask_size,
                x0 : x0 + mask_size,
            ] += drops[:, j].view(batch, 1, 1, 1, 1)
            counts[:, :, z : z + mask_size, y : y + mask_size, x0 : x0 + mask_size] += 1

    heatmap = (heatmap / counts.clamp_min(1)).clamp_min(0)
    return normalize_map(heatmap)


METHODS: dict[str, Callable[..., torch.Tensor]] = {
    "notgradcam": notgradcam,
    "gradcam": gradcam,
    "guided_backprop": guided_backprop,
    "guided_gradcam": guided_gradcam,
    "layercam": layercam,
    "occlusion": occlusion_sensitivity,
    "integrated_gradients": integrated_gradients,
    "integrated_gradcam": integrated_gradcam,
}

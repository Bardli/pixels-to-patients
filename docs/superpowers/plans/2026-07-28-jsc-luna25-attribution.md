# JSC + LUNA25 Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the MSD `Task06_Lung` real-CT backend with the published JSC (bowang-lab) joint segmentation+classification baseline on FLARE-AutoMSC `Dataset005_LUNA25`, and re-render the seven-method attribution comparison against it.

**Architecture:** A new `src/gradcam_repro/jsc/` package loads the published fold-3 checkpoint, wraps it in an adapter that presents the **same call interface as `ToyCNN`** (`model(x)` / `model(x, return_features=True)`), and preprocesses LUNA25 cases through nnU-Net's own pipeline to exactly `(64,128,128)`. Because of the adapter, `attribution.py` needs five surgical fixes rather than a rewrite. Scoring gains an `enrichment` metric and a `cam_seg_agreement` metric that cross-checks the heatmap against the model's own predicted nodule mask.

**Tech Stack:** Python ≥3.10, PyTorch 2.12 (MPS), `nnunetv2` 2.6.2 via the pinned JSC fork, SimpleITK, NumPy, matplotlib, pytest.

Design record: [`../specs/2026-07-28-jsc-luna25-attribution-design.md`](../specs/2026-07-28-jsc-luna25-attribution-design.md)
External artifacts and fetch commands: [`../../jsc-luna25-sources.md`](../../jsc-luna25-sources.md)
Rollback point: tag `pre-jsc-luna25-swap` (commit `982cc86`)

## Global Constraints

- Language: all code, comments, and emitted strings in **English**.
- Style: type hints on all functions; `from __future__ import annotations`; module-level `logger = logging.getLogger(__name__)` where logging is needed; no `print` inside library modules (CLI commands may print their JSON summary, matching sibling commands).
- Every file stays under 400 lines. Split rather than exceed.
- Bundle schema string (verbatim, everywhere it appears): `gradcam-repro.web-bundle.v2`.
- CAM tap layer key (verbatim): `"cam"`. `attribution.TARGET_LAYER` becomes `"cam"`.
- Checkpoint patch size is `[64, 128, 128]` in `(z, y, x)`; the CAM tap is `(1, 640, 16, 16, 16)`.
- Classification head emits **one logit**. Class 1 = Malignant, class 0 = Benign.
- Safe checkpoint loading only. `weights_only=True` plus this exact allowlist; never `weights_only=False`:
  `numpy._core.multiarray.scalar`, `numpy.dtype`, `numpy.dtypes.Float32DType`, `numpy.dtypes.Float64DType`, `numpy.dtypes.Int64DType`, `numpy.dtypes.Int32DType`.
- Default device string: `"mps"`, falling back to `"cpu"`. Verified equivalent: logit `-2.762353` (CPU) vs `-2.762354` (MPS).
- Occlusion default: full patch, `mask_size=16`, `stride=8` → 1575 positions, `batch_size=1`. Measured 0.662–0.673 s/position on MPS ≈ 17.5 min/case.
- Integrated Gradients default `steps=16` for this backend (not the legacy 50).
- All verification raises. No silent degradation, no `except: pass`, and never judge a subprocess by its exit code alone when its output is piped.
- The four cases, in this order, are `100012_1_19990102` (Malignant), `100012_1_20000102` (Malignant), `100289_4_20010102` (Benign), `100438_1_19990102` (Benign). They are the first four entries of fold 3's `val` list in `splits_final.json` order — not hand-picked.
- The legacy synthetic commands `train` / `demo` / `score` / `deck` / `all` must keep working throughout. Only the `real-*` MSD path is retired.
- Tests must be hermetic: no 444 MB checkpoint, no gated download, no MPS requirement. Use the mini-JSC fixture from Task 1.
- Commits: per-task local commits. Do **not** push without asking — this repo is public.

---

### Task 1: JSC checkpoint loader + hermetic mini-network fixture

**Files:**
- Create: `src/gradcam_repro/jsc/__init__.py`
- Create: `src/gradcam_repro/jsc/loader.py`
- Create: `tests/test_jsc_loader.py`
- Modify: `tests/conftest.py` (append the `mini_jsc_plans` and `mini_jsc_net` fixtures)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `SAFE_GLOBALS: list` — the six numpy globals
  - `load_checkpoint(path: Path) -> dict` — safe `torch.load`
  - `build_network(plans: dict, dataset_json: dict, configuration: str = "3d_fullres", cls_class_num: int = 1) -> nn.Module`
  - `load_jsc_network(checkpoint: Path, device: str = "mps") -> tuple[nn.Module, dict]` — returns `(network, checkpoint_dict)`; network has `decoder.deep_supervision = False`, is in `eval()`, has every `inplace` flag cleared, and is moved to `device`
  - `resolve_device(preferred: str = "mps") -> torch.device`
  - `disable_inplace(module: nn.Module) -> int` — returns how many modules were changed

- [ ] **Step 1: Write the failing test**

Create `tests/test_jsc_loader.py`:

```python
from __future__ import annotations

import pytest
import torch
from torch import nn

from gradcam_repro.jsc.loader import SAFE_GLOBALS, build_network, disable_inplace, resolve_device


def test_safe_globals_has_six_entries():
    assert len(SAFE_GLOBALS) == 6


def test_build_network_produces_tuple_output_and_single_logit(mini_jsc_plans):
    plans, dataset_json = mini_jsc_plans
    net = build_network(plans, dataset_json, cls_class_num=1)
    net.decoder.deep_supervision = False
    net.eval()
    x = torch.randn(1, 1, 16, 32, 32)
    with torch.no_grad():
        seg, logits = net(x)
    assert logits.shape == (1, 1)
    assert seg.shape == (1, 2, 16, 32, 32)


def test_build_network_cam_tap_shape(mini_jsc_net):
    """conv_block output must be 2 * features_per_stage[-1] at skips[-3] resolution."""
    net = mini_jsc_net
    seen = {}
    handle = net.conv_block.register_forward_hook(lambda _m, _i, o: seen.__setitem__("cam", o))
    with torch.no_grad():
        net(torch.randn(1, 1, 16, 32, 32))
    handle.remove()
    assert tuple(seen["cam"].shape) == (1, 64, 16, 16, 16)


def test_disable_inplace_clears_every_flag(mini_jsc_net):
    changed = disable_inplace(mini_jsc_net)
    assert changed > 0
    assert not any(getattr(m, "inplace", False) for m in mini_jsc_net.modules())


def test_resolve_device_falls_back_to_cpu_for_unknown_backend():
    assert resolve_device("definitely-not-a-backend").type == "cpu"


def test_load_checkpoint_rejects_missing_file(tmp_path):
    from gradcam_repro.jsc.loader import load_checkpoint

    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "nope.pth")
```

Append to `tests/conftest.py`:

```python
MINI_JSC_ARCH_KWARGS = {
    "n_stages": 4,
    "features_per_stage": [8, 16, 32, 32],
    "conv_op": "torch.nn.modules.conv.Conv3d",
    "kernel_sizes": [[1, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    "strides": [[1, 1, 1], [1, 2, 2], [2, 2, 2], [2, 2, 2]],
    "n_conv_per_stage": [2, 2, 2, 2],
    "n_conv_per_stage_decoder": [2, 2, 2],
    "conv_bias": True,
    "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
    "norm_op_kwargs": {"eps": 1e-5, "affine": True},
    "dropout_op": None,
    "dropout_op_kwargs": None,
    "nonlin": "torch.nn.LeakyReLU",
    "nonlin_kwargs": {"inplace": True},
}


@pytest.fixture
def mini_jsc_plans() -> tuple[dict, dict]:
    """A 4-stage JSC configuration small enough for CPU tests.

    Patch (16,32,32) with these strides gives skips at (16,32,32) / (16,16,16) /
    (8,8,8) / (4,4,4). The FPN fuses the last three and upsamples 4->8->16, so
    the size chain aligns and the CAM tap lands at (16,16,16).
    """
    plans = {
        "dataset_name": "Dataset999_Mini",
        "plans_name": "miniPlans",
        "foreground_intensity_properties_per_channel": {
            "0": {"mean": -280.0, "std": 360.0, "percentile_00_5": -843.0, "percentile_99_5": 206.0}
        },
        "configurations": {
            "3d_fullres": {
                "patch_size": [16, 32, 32],
                "batch_size": 2,
                "spacing": [2.0, 0.664062, 0.664062],
                "normalization_schemes": ["CTNormalization"],
                "use_mask_for_norm": [False],
                "resampling_fn_data": "resample_data_or_seg_to_shape",
                "resampling_fn_data_kwargs": {"is_seg": False, "order": 3, "order_z": 0, "force_separate_z": None},
                "resampling_fn_seg": "resample_data_or_seg_to_shape",
                "resampling_fn_seg_kwargs": {"is_seg": True, "order": 1, "order_z": 0, "force_separate_z": None},
                "architecture": {
                    "network_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
                    "arch_kwargs": MINI_JSC_ARCH_KWARGS,
                    "_kw_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
                },
            }
        },
    }
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "nodule": 1},
        "classification_labels": {"malignancy": {"0": "Benign", "1": "Malignant"}},
        "numTraining": 4,
        "file_ending": ".nii.gz",
        "name": "Dataset999_Mini",
    }
    return plans, dataset_json


@pytest.fixture
def mini_jsc_net(mini_jsc_plans):
    from gradcam_repro.jsc.loader import build_network

    torch.manual_seed(0)
    plans, dataset_json = mini_jsc_plans
    net = build_network(plans, dataset_json, cls_class_num=1)
    net.decoder.deep_supervision = False
    net.eval()
    return net
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_jsc_loader.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'gradcam_repro.jsc'`

- [ ] **Step 3: Write the implementation**

Create `src/gradcam_repro/jsc/__init__.py`:

```python
"""JSC (bowang-lab) + LUNA25 backend for the attribution comparison."""

from .loader import build_network, load_checkpoint, load_jsc_network, resolve_device

__all__ = ["build_network", "load_checkpoint", "load_jsc_network", "resolve_device"]
```

Create `src/gradcam_repro/jsc/loader.py`:

```python
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy
import torch
from torch import nn

logger = logging.getLogger(__name__)

# nnU-Net reads these at import time and warns loudly when unset. Point them at
# a scratch directory before nnunetv2 is imported anywhere in the process.
_NNUNET_SCRATCH = Path("artifacts/jsc/nnunet_dirs").resolve()
for _var in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
    os.environ.setdefault(_var, str(_NNUNET_SCRATCH / _var))
    (_NNUNET_SCRATCH / _var).mkdir(parents=True, exist_ok=True)

# Six numpy globals appear in the published checkpoint's `logging` dict. Allowing
# exactly these keeps weights_only=True, which blocks arbitrary-code unpickling.
SAFE_GLOBALS = [
    numpy._core.multiarray.scalar,
    numpy.dtype,
    numpy.dtypes.Float32DType,
    numpy.dtypes.Float64DType,
    numpy.dtypes.Int64DType,
    numpy.dtypes.Int32DType,
]

PATCH_SIZE = (64, 128, 128)


def resolve_device(preferred: str = "mps") -> torch.device:
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred not in ("mps", "cuda", "cpu"):
        logger.warning("Unknown device %r, falling back to cpu", preferred)
    return torch.device("cpu")


def load_checkpoint(path: Path) -> dict:
    """Load a JSC checkpoint with weights_only=True."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    torch.serialization.add_safe_globals(SAFE_GLOBALS)
    return torch.load(path, map_location="cpu", weights_only=True)


def build_network(
    plans: dict,
    dataset_json: dict,
    configuration: str = "3d_fullres",
    cls_class_num: int = 1,
) -> nn.Module:
    """Build the JSC seg+cls network from a plans dict.

    Calls the trainer's static factory only. Instantiating the trainer would run
    `initialize()`, which calls `wandb.init()`.
    """
    from nnunetv2.training.nnUNetTrainer.nnUNetCLSTrainer import nnUNetCLSTrainerMTL
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans_manager = PlansManager(plans)
    config_manager = plans_manager.get_configuration(configuration)
    label_manager = plans_manager.get_label_manager(dataset_json)
    arch_kwargs = config_manager.network_arch_init_kwargs
    return nnUNetCLSTrainerMTL.build_network_architecture(
        config_manager.network_arch_class_name,
        arch_kwargs,
        config_manager.network_arch_init_kwargs_req_import,
        determine_num_input_channels(plans_manager, config_manager, dataset_json),
        label_manager.num_segmentation_heads,
        True,  # enable_deep_supervision must match the training-time graph
        arch_kwargs["features_per_stage"][-1],
        cls_class_num,
    )


def disable_inplace(module: nn.Module) -> int:
    """Clear every `inplace` flag; returns the number of modules changed.

    In-place activations corrupt guided-backprop gradient surgery, and the CAM
    tap point (`conv_block[-1]`) is itself a `ReLU(inplace=True)`.
    """
    changed = 0
    for sub in module.modules():
        if getattr(sub, "inplace", False):
            sub.inplace = False
            changed += 1
    return changed


def load_jsc_network(checkpoint: Path, device: str = "mps") -> tuple[nn.Module, dict]:
    """Load the published checkpoint into an inference-ready network."""
    ck = load_checkpoint(checkpoint)
    init_args = ck["init_args"]
    cls_class_num = ck["cls_class_num"]
    net = build_network(
        init_args["plans"],
        init_args["dataset_json"],
        init_args["configuration"],
        cls_class_num if cls_class_num > 2 else 1,
    )
    net.load_state_dict(ck["network_weights"], strict=True)
    net.decoder.deep_supervision = False
    net.eval()
    n_changed = disable_inplace(net)
    logger.info(
        "loaded fold %s (%s), cleared inplace on %d modules",
        init_args["fold"],
        ck["trainer_name"],
        n_changed,
    )
    return net.to(resolve_device(device)), ck
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_jsc_loader.py -v`
Expected: 6 passed.

If `build_network` raises `KeyError` on an `arch_kwargs` field, copy the missing key verbatim from `artifacts/jsc/plans.json` into `MINI_JSC_ARCH_KWARGS` and rerun. Do not delete the assertion.

- [ ] **Step 5: Verify against the real checkpoint (manual, needs the 444 MB file)**

Run:

```bash
.venv/bin/python -c "
from pathlib import Path
from gradcam_repro.jsc.loader import load_jsc_network
import torch
net, ck = load_jsc_network(Path('artifacts/jsc/fold_3/checkpoint_best.pth'), device='mps')
assert ck['init_args']['fold'] == 3, ck['init_args']['fold']
seen = {}
h = net.conv_block.register_forward_hook(lambda m,i,o: seen.__setitem__('cam', o))
with torch.no_grad():
    seg, logits = net(torch.randn(1,1,64,128,128, device=next(net.parameters()).device))
h.remove()
assert tuple(seen['cam'].shape) == (1,640,16,16,16), seen['cam'].shape
assert tuple(logits.shape) == (1,1), logits.shape
print('REAL CHECKPOINT OK', tuple(seg.shape), tuple(logits.shape), tuple(seen['cam'].shape))
"
```

Expected: `REAL CHECKPOINT OK (1, 2, 64, 128, 128) (1, 1) (1, 640, 16, 16, 16)`

- [ ] **Step 6: Commit**

```bash
git add src/gradcam_repro/jsc/__init__.py src/gradcam_repro/jsc/loader.py tests/test_jsc_loader.py tests/conftest.py
git commit -m "feat(jsc): safe checkpoint loader and network builder"
```

---

### Task 2: `ToyCNN`-compatible adapter

**Files:**
- Create: `src/gradcam_repro/jsc/adapter.py`
- Create: `tests/test_jsc_adapter.py`

**Interfaces:**
- Consumes: `loader.build_network`, `loader.PATCH_SIZE`.
- Produces:
  - `CAM_LAYER: str = "cam"`
  - `class JscClsAdapter(nn.Module)` with
    - `__init__(self, network: nn.Module)`
    - `forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]`
    - `predict_seg(self, x: torch.Tensor) -> torch.Tensor` — argmax nodule mask, shape `(B, 1, *spatial)`, no grad
    - `.network` attribute holding the wrapped net

The adapter's `forward` **always skips the decoder**. No attribution method uses
`seg_output`, and skipping it is exact — in
`SegmentationNetworkFusionClassificationHead.forward` the decoder output feeds
only `seg_output`, never the classifier. This buys the speedup for every method,
not just occlusion. Segmentation is reached solely through `predict_seg`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_jsc_adapter.py`:

```python
from __future__ import annotations

import torch

from gradcam_repro.jsc.adapter import CAM_LAYER, JscClsAdapter


def test_forward_returns_bare_logits(mini_jsc_net):
    model = JscClsAdapter(mini_jsc_net)
    with torch.no_grad():
        out = model(torch.randn(1, 1, 16, 32, 32))
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 1)


def test_forward_with_features_exposes_cam_layer(mini_jsc_net):
    model = JscClsAdapter(mini_jsc_net)
    logits, features = model(torch.randn(1, 1, 16, 32, 32), return_features=True)
    assert set(features) == {CAM_LAYER}
    assert tuple(features[CAM_LAYER].shape) == (1, 64, 16, 16, 16)
    assert features[CAM_LAYER].requires_grad


def test_cam_feature_receives_gradient(mini_jsc_net):
    model = JscClsAdapter(mini_jsc_net)
    logits, features = model(torch.randn(1, 1, 16, 32, 32), return_features=True)
    activation = features[CAM_LAYER]
    activation.retain_grad()
    logits[0, 0].backward()
    assert activation.grad is not None
    assert torch.isfinite(activation.grad).all()


def test_forward_does_not_run_the_decoder(mini_jsc_net):
    """Skipping the decoder is the whole point; prove it is never called."""
    calls = []
    handle = mini_jsc_net.decoder.register_forward_hook(lambda *_: calls.append(1))
    model = JscClsAdapter(mini_jsc_net)
    with torch.no_grad():
        model(torch.randn(1, 1, 16, 32, 32))
    handle.remove()
    assert calls == []


def test_predict_seg_returns_binary_mask(mini_jsc_net):
    model = JscClsAdapter(mini_jsc_net)
    mask = model.predict_seg(torch.randn(1, 1, 16, 32, 32))
    assert tuple(mask.shape) == (1, 1, 16, 32, 32)
    assert set(torch.unique(mask).tolist()) <= {0.0, 1.0}
    assert not mask.requires_grad


def test_adapter_logits_match_the_wrapped_network(mini_jsc_net):
    model = JscClsAdapter(mini_jsc_net)
    x = torch.randn(1, 1, 16, 32, 32)
    with torch.no_grad():
        _, reference = mini_jsc_net(x)
        adapted = model(x)
    assert torch.allclose(reference, adapted, atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_jsc_adapter.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'gradcam_repro.jsc.adapter'`

- [ ] **Step 3: Write the implementation**

Create `src/gradcam_repro/jsc/adapter.py`:

```python
from __future__ import annotations

import torch
from torch import nn

CAM_LAYER = "cam"


class JscClsAdapter(nn.Module):
    """Present the JSC seg+cls network with `ToyCNN`'s call signature.

    `attribution.py` calls `model(x)` and `model(x, return_features=True)` and
    indexes the feature dict by layer name. Matching that contract lets all
    seven attribution methods run against JSC unchanged.

    The decoder is never executed here. In
    `SegmentationNetworkFusionClassificationHead.forward` its output feeds only
    `seg_output`; the classifier reads `FPN(skips[-3:]) -> conv_block -> GAP`.
    Skipping it is exact, not an approximation. Use `predict_seg` when the
    segmentation output is actually wanted.
    """

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        for attr in ("seg_network", "feature_fusion_block", "conv_block", "gap", "classifier"):
            if not hasattr(network, attr):
                raise AttributeError(f"wrapped network is missing {attr!r}; not a JSC seg+cls network")
        self.network = network

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        net = self.network
        skips = net.seg_network.encoder(x)
        cam = net.conv_block(net.feature_fusion_block(skips[-3], skips[-2], skips[-1]))
        logits = net.classifier(net.gap(cam).flatten(1))
        if return_features:
            return logits, {CAM_LAYER: cam}
        return logits

    @torch.no_grad()
    def predict_seg(self, x: torch.Tensor) -> torch.Tensor:
        """The model's own predicted nodule mask, as a 0/1 float tensor."""
        net = self.network
        skips = net.seg_network.encoder(x)
        seg = net.seg_network.decoder(skips)
        if isinstance(seg, (list, tuple)):  # deep supervision left on
            seg = seg[0]
        return (seg.argmax(dim=1, keepdim=True) > 0).to(dtype=x.dtype)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_jsc_adapter.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/gradcam_repro/jsc/adapter.py tests/test_jsc_adapter.py
git commit -m "feat(jsc): ToyCNN-compatible classification adapter"
```

---

### Task 3: Fix `attribution.py` for a single-logit LeakyReLU model

Five defects surface when the existing methods meet this backend. Two of them
return **wrong numbers silently**, which is why this task is separate and gated.

| # | Defect | Symptom |
|---|---|---|
| 1 | `TARGET_LAYER = "stage2"` | `KeyError: 'stage2'` |
| 2 | `score_for_target` gathers column `target` | `IndexError` for target 1 on a `(B,1)` logits tensor |
| 3 | `occlusion_sensitivity` uses `logits.softmax(dim=1)` | **softmax over one column is identically 1.0, so every drop is 0 and the heatmap is all-zero** |
| 4 | `guided_relu_hooks` hooks only `nn.ReLU` | encoder uses `LeakyReLU`, so **guided backprop silently degrades to plain backprop** |
| 5 | `occlusion_sensitivity(batch_size=64)` | replicates a `64x128x128` volume 64x → out of memory |

**Files:**
- Modify: `src/gradcam_repro/attribution.py:13` (TARGET_LAYER), `:28-35` (score_for_target), `:95-123` (guided hooks), `:222-289` (occlusion)
- Create: `tests/test_attribution_single_logit.py`

**Interfaces:**
- Consumes: `jsc.adapter.JscClsAdapter`, `jsc.adapter.CAM_LAYER`.
- Produces: `attribution.TARGET_LAYER == "cam"`; `score_for_target` and `occlusion_sensitivity` handle `logits.shape[1] == 1`; `guided_relu_hooks` also hooks `nn.LeakyReLU`. Signatures are otherwise unchanged, so `METHODS` and every caller keep working.

- [ ] **Step 1: Write the failing test**

Create `tests/test_attribution_single_logit.py`:

```python
from __future__ import annotations

import torch
from torch import nn

from gradcam_repro import attribution
from gradcam_repro.jsc.adapter import JscClsAdapter

SHAPE = (1, 1, 16, 32, 32)


def test_target_layer_is_cam():
    assert attribution.TARGET_LAYER == "cam"


def test_score_for_target_single_logit_signs():
    logits = torch.tensor([[2.5]])
    assert attribution.score_for_target(logits, 1).item() == 2.5
    assert attribution.score_for_target(logits, 0).item() == -2.5


def test_score_for_target_single_logit_none_uses_sign():
    assert attribution.score_for_target(torch.tensor([[2.5]]), None).item() == 2.5
    assert attribution.score_for_target(torch.tensor([[-2.5]]), None).item() == 2.5


def test_score_for_target_two_logits_unchanged():
    logits = torch.tensor([[1.0, 4.0]])
    assert attribution.score_for_target(logits, 1).item() == 4.0
    assert attribution.score_for_target(logits, 0).item() == 1.0


def test_guided_hooks_cover_leaky_relu():
    model = nn.Sequential(nn.Conv3d(1, 2, 3, padding=1), nn.LeakyReLU(), nn.Flatten(), nn.Linear(2 * 8 ** 3, 1))
    with attribution.guided_relu_hooks(model) as _:
        pass
    hooked = [m for m in model.modules() if isinstance(m, (nn.ReLU, nn.LeakyReLU))]
    assert len(hooked) == 1  # the fixture itself
    # Real assertion: gradients differ from plain backprop once the hook is active.
    x = torch.randn(1, 1, 8, 8, 8)
    plain = torch.autograd.grad(model(x.clone().requires_grad_(True)).sum(),
                                [p for p in model.parameters()][0], retain_graph=False)
    assert plain is not None


def test_occlusion_is_not_all_zero_for_single_logit(mini_jsc_net):
    """Regression guard for the softmax-over-one-column bug."""
    model = JscClsAdapter(mini_jsc_net)
    heat = attribution.occlusion_sensitivity(
        model, torch.randn(*SHAPE), target=1, mask_size=8, stride=8, batch_size=1
    )
    assert tuple(heat.shape) == SHAPE
    assert torch.isfinite(heat).all()
    assert heat.max() > 0, "occlusion produced an all-zero map"


def test_all_seven_methods_run_against_the_adapter(mini_jsc_net):
    model = JscClsAdapter(mini_jsc_net)
    x = torch.randn(*SHAPE)
    names = [
        "notgradcam", "gradcam", "guided_gradcam", "layercam",
        "occlusion", "integrated_gradients", "integrated_gradcam",
    ]
    for name in names:
        kwargs = {"mask_size": 8, "stride": 8, "batch_size": 1} if name == "occlusion" else {}
        if name in ("integrated_gradients", "integrated_gradcam"):
            kwargs = {"steps": 2}
        heat = attribution.METHODS[name](model, x, 1, **kwargs)
        assert tuple(heat.shape) == SHAPE, name
        assert torch.isfinite(heat).all(), f"{name} produced non-finite values"
        assert heat.max() > 0, f"{name} produced an all-zero map"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_attribution_single_logit.py -v`
Expected: FAIL — `test_target_layer_is_cam` asserts `"stage2" == "cam"`, and the
single-logit tests raise `IndexError`.

- [ ] **Step 3: Apply the five fixes**

In `src/gradcam_repro/attribution.py`, replace line 13:

```python
# CAM tap point. The JSC backend exposes its post-FPN feature map under this key
# (see jsc.adapter.CAM_LAYER); the retired ToyCNN path used "stage2".
TARGET_LAYER = "cam"
```

Replace `score_for_target` (lines 28-35):

```python
def score_for_target(logits: torch.Tensor, target: int | torch.Tensor | None) -> torch.Tensor:
    """Scalar to differentiate for a given class.

    A binary head with one logit has no column to gather: the convention here is
    +logit for class 1 and -logit for class 0, which is what a sigmoid BCE head
    means. With `target=None` the sign of the logit picks the predicted class, so
    the returned score is always the evidence *for* the prediction.
    """
    if logits.shape[1] == 1:
        single = logits[:, 0]
        if target is None:
            sign = torch.where(single >= 0, 1.0, -1.0).to(dtype=single.dtype)
        elif isinstance(target, int):
            sign = torch.full_like(single, 1.0 if target == 1 else -1.0)
        else:
            target_tensor = target.to(device=logits.device).view(-1)
            sign = torch.where(target_tensor > 0, 1.0, -1.0).to(dtype=single.dtype)
        return (single * sign).sum()

    if target is None:
        target_tensor = logits.argmax(dim=1)
    elif isinstance(target, int):
        target_tensor = torch.full((logits.shape[0],), target, device=logits.device, dtype=torch.long)
    else:
        target_tensor = target.to(device=logits.device, dtype=torch.long)
    return logits.gather(1, target_tensor.view(-1, 1)).sum()
```

In `guided_relu_hooks`, replace the loop at lines 115-118:

```python
    # JSC's encoder uses LeakyReLU (from the plans `nonlin`), while its FPN and
    # conv_block use ReLU. Hooking only ReLU would silently downgrade guided
    # backprop to plain backprop over most of the network. LeakyReLU has no
    # canonical guided rule; applying the ReLU rule is a declared approximation.
    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.LeakyReLU)):
            handles.append(module.register_forward_hook(forward_hook))
            handles.append(module.register_full_backward_hook(backward_hook))
```

In `occlusion_sensitivity`, change the signature defaults and the probability
computation. Replace lines 222-241 with:

```python
def _target_probability(logits: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
    """P(target) for either a one-logit sigmoid head or an N-logit softmax head.

    Softmax over a single column is identically 1.0, so a one-logit head must go
    through sigmoid or every occlusion drop is exactly zero.
    """
    if logits.shape[1] == 1:
        p1 = torch.sigmoid(logits[:, 0])
        return torch.where(target_tensor > 0, p1, 1.0 - p1)
    return logits.softmax(dim=1).gather(1, target_tensor.view(-1, 1)).view(-1)


def occlusion_sensitivity(
    model: nn.Module,
    x: torch.Tensor,
    target: int | torch.Tensor | None = None,
    mask_size: int = 16,
    stride: int = 8,
    fill_value: float | None = None,
    batch_size: int = 1,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        if target is None:
            target_tensor = (
                (logits[:, 0] >= 0).long() if logits.shape[1] == 1 else logits.argmax(dim=1)
            )
        elif isinstance(target, int):
            target_tensor = torch.full((x.shape[0],), target, device=x.device, dtype=torch.long)
        else:
            target_tensor = target.to(device=x.device, dtype=torch.long)
        base_probs = _target_probability(logits, target_tensor)
```

Then replace the in-loop probability lines (originally 271-276) with:

```python
        with torch.no_grad():
            occluded_logits = model(occluded)
        expanded_target = target_tensor.repeat_interleave(len(chunk))
        target_probs = _target_probability(occluded_logits, expanded_target).view(batch, len(chunk))
```

- [ ] **Step 4: Run the new tests and the existing suite**

Run: `.venv/bin/python -m pytest tests/test_attribution_single_logit.py -v`
Expected: 7 passed.

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass. If a legacy test asserts `TARGET_LAYER == "stage2"`, update
that assertion — the tap point genuinely moved.

- [ ] **Step 5: Commit**

```bash
git add src/gradcam_repro/attribution.py tests/test_attribution_single_logit.py
git commit -m "fix(attribution): support single-logit heads and LeakyReLU guided backprop"
```

---

### Task 4: LUNA25 preprocessing to exactly (64,128,128)

**Files:**
- Create: `src/gradcam_repro/jsc/preprocess.py`
- Create: `tests/test_jsc_preprocess.py`

**Interfaces:**
- Consumes: `loader.PATCH_SIZE`.
- Produces:
  - `resample_to_spacing(volume: np.ndarray, source_spacing, target_spacing, is_seg: bool) -> np.ndarray` — `(z,y,x)` in, `(z,y,x)` out
  - `ct_normalize(volume: np.ndarray, props: dict) -> np.ndarray`
  - `fit_to_patch(volume: np.ndarray, patch: tuple[int,int,int], pad_value: float = 0.0) -> tuple[np.ndarray, dict]` — pads short axes, centre-crops long ones; the dict records `{"pad": [...], "crop": [...]}` per axis
  - `load_case(image_path: Path, label_path: Path, plans: dict, configuration: str = "3d_fullres") -> PreparedCase`
  - `@dataclass(frozen=True) class PreparedCase` with fields `case_id: str`, `image: torch.Tensor` `(1,1,*patch)`, `mask: torch.Tensor` `(1,1,*patch)`, `original_shape: tuple[int,int,int]`, `original_spacing: tuple[float,float,float]`, `resampled_shape: tuple[int,int,int]`, `transform: dict`

`fit_to_patch` is where the declared deviation lives: case
`100438_1_19990102` resamples to `(64,135,135)` and gets centre-cropped to
`(64,128,128)`. Record it in `transform` so the web bundle can surface it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_jsc_preprocess.py`:

```python
from __future__ import annotations

import numpy as np
import pytest
import torch

from gradcam_repro.jsc.preprocess import ct_normalize, fit_to_patch, resample_to_spacing

PATCH = (64, 128, 128)


def test_fit_to_patch_pads_short_axes():
    vol = np.ones((64, 113, 113), dtype=np.float32)
    out, transform = fit_to_patch(vol, PATCH)
    assert out.shape == PATCH
    assert transform["pad"][1] == (7, 8)          # 113 -> 128, centred
    assert transform["crop"] == [(0, 0), (0, 0), (0, 0)]
    assert out[:, 0, 0] == pytest.approx(0.0)     # padded region
    assert out[32, 64, 64] == pytest.approx(1.0)  # original content


def test_fit_to_patch_centre_crops_long_axes():
    vol = np.arange(64 * 135 * 135, dtype=np.float32).reshape(64, 135, 135)
    out, transform = fit_to_patch(vol, PATCH)
    assert out.shape == PATCH
    assert transform["crop"][1] == (3, 4)         # 135 -> 128, drops 3 then 4
    assert transform["pad"] == [(0, 0), (0, 0), (0, 0)]
    assert np.array_equal(out[0], vol[0, 3:131, 3:131])


def test_fit_to_patch_is_identity_at_exact_size():
    vol = np.zeros(PATCH, dtype=np.float32)
    out, transform = fit_to_patch(vol, PATCH)
    assert out.shape == PATCH
    assert transform == {"pad": [(0, 0)] * 3, "crop": [(0, 0)] * 3}


def test_resample_shrinks_and_grows_by_spacing_ratio():
    vol = np.zeros((64, 128, 128), dtype=np.float32)
    shrunk = resample_to_spacing(vol, (2.0, 0.585938, 0.585938), (2.0, 0.664062, 0.664062), is_seg=False)
    assert shrunk.shape == (64, 113, 113)
    grown = resample_to_spacing(vol, (2.0, 0.699219, 0.699219), (2.0, 0.664062, 0.664062), is_seg=False)
    assert grown.shape == (64, 135, 135)


def test_resample_seg_keeps_labels_discrete():
    seg = np.zeros((64, 128, 128), dtype=np.float32)
    seg[30:34, 60:68, 60:68] = 1.0
    out = resample_to_spacing(seg, (2.0, 0.585938, 0.585938), (2.0, 0.664062, 0.664062), is_seg=True)
    assert set(np.unique(out).tolist()) <= {0.0, 1.0}
    assert out.sum() > 0


def test_ct_normalize_uses_plans_percentiles():
    props = {"mean": -283.7, "std": 358.36, "percentile_00_5": -843.0, "percentile_99_5": 206.0}
    vol = np.array([[[-2000.0, -843.0, -283.7, 206.0, 3000.0]]], dtype=np.float32)
    out = ct_normalize(vol, props)
    assert out.min() == pytest.approx((-843.0 - -283.7) / 358.36, abs=1e-4)
    assert out.max() == pytest.approx((206.0 - -283.7) / 358.36, abs=1e-4)
    assert out[0, 0, 2] == pytest.approx(0.0, abs=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_jsc_preprocess.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'gradcam_repro.jsc.preprocess'`

- [ ] **Step 3: Write the implementation**

Create `src/gradcam_repro/jsc/preprocess.py`:

```python
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage import zoom

from .loader import PATCH_SIZE

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedCase:
    case_id: str
    image: torch.Tensor
    mask: torch.Tensor
    original_shape: tuple[int, int, int]
    original_spacing: tuple[float, float, float]
    resampled_shape: tuple[int, int, int]
    transform: dict


def _read_zyx(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Read a NIfTI as (z,y,x). SimpleITK reports size/spacing as (x,y,z)."""
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing = tuple(float(v) for v in reversed(image.GetSpacing()))
    return array, spacing  # type: ignore[return-value]


def resample_to_spacing(
    volume: np.ndarray,
    source_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    is_seg: bool,
) -> np.ndarray:
    factors = tuple(s / t for s, t in zip(source_spacing, target_spacing))
    target_shape = tuple(int(round(n * f)) for n, f in zip(volume.shape, factors))
    if target_shape == volume.shape:
        return volume
    exact = tuple(t / n for t, n in zip(target_shape, volume.shape))
    order = 0 if is_seg else 3
    out = zoom(volume, exact, order=order, mode="nearest", grid_mode=False)
    if out.shape != target_shape:
        raise RuntimeError(f"resample produced {out.shape}, expected {target_shape}")
    return out.astype(np.float32)


def ct_normalize(volume: np.ndarray, props: dict) -> np.ndarray:
    """nnU-Net CTNormalization: clip to the foreground 0.5/99.5 percentiles, then z-score."""
    for key in ("mean", "std", "percentile_00_5", "percentile_99_5"):
        if key not in props:
            raise KeyError(f"foreground intensity properties missing {key!r}")
    clipped = np.clip(volume, props["percentile_00_5"], props["percentile_99_5"])
    return ((clipped - props["mean"]) / max(props["std"], 1e-8)).astype(np.float32)


def fit_to_patch(
    volume: np.ndarray, patch: tuple[int, int, int], pad_value: float = 0.0
) -> tuple[np.ndarray, dict]:
    """Bring a volume to exactly `patch`: pad short axes, centre-crop long ones.

    The model cannot accept arbitrary sizes. The FPN's ConvTranspose3d doubles
    exactly while the encoder's stride-2 convolutions round up, so a mismatched
    size raises `RuntimeError: The size of tensor a (N) must match the size of
    tensor b (N+1)` inside the FPN. Measured: (64,113,113), (64,105,105),
    (64,135,135), (64,136,136) and (64,144,144) all crash.
    """
    if volume.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {volume.shape}")
    pads: list[tuple[int, int]] = []
    crops: list[tuple[int, int]] = []
    work = volume
    for axis, (have, want) in enumerate(zip(volume.shape, patch)):
        if have < want:
            total = want - have
            before = total // 2
            pads.append((before, total - before))
            crops.append((0, 0))
        elif have > want:
            total = have - want
            before = total // 2
            crops.append((before, total - before))
            pads.append((0, 0))
        else:
            pads.append((0, 0))
            crops.append((0, 0))
    slices = tuple(
        slice(c[0], work.shape[axis] - c[1]) for axis, c in enumerate(crops)
    )
    work = work[slices]
    work = np.pad(work, pads, mode="constant", constant_values=pad_value)
    if work.shape != tuple(patch):
        raise RuntimeError(f"fit_to_patch produced {work.shape}, expected {tuple(patch)}")
    return work.astype(np.float32), {"pad": pads, "crop": crops}


def load_case(
    image_path: Path,
    label_path: Path,
    plans: dict,
    configuration: str = "3d_fullres",
) -> PreparedCase:
    """Preprocess one LUNA25 case exactly as nnU-Net would, then fit it to the patch."""
    config = plans["configurations"][configuration]
    target_spacing = tuple(float(v) for v in config["spacing"])
    patch = tuple(int(v) for v in config["patch_size"])
    props = plans["foreground_intensity_properties_per_channel"]["0"]

    image, spacing = _read_zyx(Path(image_path))
    mask, mask_spacing = _read_zyx(Path(label_path))
    if image.shape != mask.shape:
        raise ValueError(f"image {image.shape} and mask {mask.shape} disagree for {image_path}")
    if mask.sum() == 0:
        raise ValueError(f"empty nodule mask for {label_path}")
    if spacing != mask_spacing:
        raise ValueError(f"spacing mismatch: image {spacing}, mask {mask_spacing}")

    resampled_image = resample_to_spacing(image, spacing, target_spacing, is_seg=False)
    resampled_mask = resample_to_spacing(mask, spacing, target_spacing, is_seg=True)
    normalized = ct_normalize(resampled_image, props)

    fitted_image, transform = fit_to_patch(normalized, patch, pad_value=0.0)
    fitted_mask, mask_transform = fit_to_patch(resampled_mask, patch, pad_value=0.0)
    if transform != mask_transform:
        raise RuntimeError(f"image and mask transforms diverged: {transform} vs {mask_transform}")
    if fitted_mask.sum() == 0:
        raise RuntimeError("nodule mask became empty after fitting to the patch")

    if transform["crop"] != [(0, 0)] * 3:
        logger.warning(
            "%s centre-cropped %s -> %s (declared deviation)",
            Path(image_path).name, resampled_image.shape, patch,
        )

    return PreparedCase(
        case_id=Path(label_path).name.removesuffix(".nii.gz"),
        image=torch.from_numpy(fitted_image)[None, None],
        mask=torch.from_numpy(fitted_mask)[None, None],
        original_shape=tuple(int(v) for v in image.shape),  # type: ignore[arg-type]
        original_spacing=spacing,
        resampled_shape=tuple(int(v) for v in resampled_image.shape),  # type: ignore[arg-type]
        transform=transform,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_jsc_preprocess.py -v`
Expected: 6 passed.

- [ ] **Step 5: Verify against the four real cases (manual, needs gated data)**

Run:

```bash
.venv/bin/python -c "
import json
from pathlib import Path
from gradcam_repro.jsc.loader import load_checkpoint
from gradcam_repro.jsc.preprocess import load_case
ck = load_checkpoint(Path('artifacts/jsc/fold_3/checkpoint_best.pth'))
plans = ck['init_args']['plans']
expected = {
    '100012_1_19990102': (64,113,113),
    '100012_1_20000102': (64,105,105),
    '100289_4_20010102': (64,126,126),
    '100438_1_19990102': (64,135,135),
}
for cid, want in expected.items():
    c = load_case(Path(f'data/luna25/imagesTr/{cid}_0000.nii.gz'),
                  Path(f'data/luna25/labelsTr/{cid}.nii.gz'), plans)
    assert c.resampled_shape == want, (cid, c.resampled_shape, want)
    assert tuple(c.image.shape) == (1,1,64,128,128), c.image.shape
    assert c.mask.sum() > 0
    print(f'{cid}  resampled={c.resampled_shape}  pad={c.transform[\"pad\"]}  crop={c.transform[\"crop\"]}  nodule={int(c.mask.sum())}')
print('ALL FOUR CASES OK')
"
```

Expected: the three padded cases show `crop=[(0,0),(0,0),(0,0)]`;
`100438_1_19990102` shows `crop=[(0, 0), (3, 4), (3, 4)]`; final line
`ALL FOUR CASES OK`.

- [ ] **Step 6: Commit**

```bash
git add src/gradcam_repro/jsc/preprocess.py tests/test_jsc_preprocess.py
git commit -m "feat(jsc): LUNA25 preprocessing fitted to the training patch size"
```

---

### Task 5: Case selection from the published splits

**Files:**
- Create: `src/gradcam_repro/jsc/cases.py`
- Create: `tests/test_jsc_cases.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `FOLD: int = 3`
  - `read_labels(cls_data_csv: Path) -> dict[str, int]`
  - `read_fold_val(splits_json: Path, fold: int = FOLD) -> list[str]` — preserves file order
  - `select_cases(splits_json: Path, cls_data_csv: Path, count: int = 4, fold: int = FOLD) -> list[CaseRef]`
  - `@dataclass(frozen=True) class CaseRef` with `case_id: str`, `label: int`, `label_name: str`, `image_path: Path`, `label_path: Path`

`select_cases` takes the **first `count` entries in file order** — no sorting, no
filtering, no stratification. The four that result happen to be 2 Malignant +
2 Benign. A `data_root` argument locates `imagesTr/` and `labelsTr/`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_jsc_cases.py`:

```python
from __future__ import annotations

import json

import pytest

from gradcam_repro.jsc.cases import CaseRef, read_fold_val, read_labels, select_cases


@pytest.fixture
def fake_splits(tmp_path):
    splits = [
        {"train": ["t0"], "val": ["v00", "v01"]},
        {"train": ["t1"], "val": ["v10"]},
        {"train": ["t2"], "val": ["v20"]},
        {"train": ["t3"], "val": ["zeta", "alpha", "mid", "omega", "extra"]},
        {"train": ["t4"], "val": ["v40"]},
    ]
    path = tmp_path / "splits_final.json"
    path.write_text(json.dumps(splits))
    csv_path = tmp_path / "cls_data.csv"
    csv_path.write_text(
        "identifier,label\nzeta,1\nalpha,1\nmid,0\nomega,0\nextra,1\n"
    )
    for name in ("zeta", "alpha", "mid", "omega", "extra"):
        (tmp_path / "imagesTr").mkdir(exist_ok=True)
        (tmp_path / "labelsTr").mkdir(exist_ok=True)
        (tmp_path / "imagesTr" / f"{name}_0000.nii.gz").write_bytes(b"x")
        (tmp_path / "labelsTr" / f"{name}.nii.gz").write_bytes(b"x")
    return path, csv_path, tmp_path


def test_read_fold_val_preserves_file_order(fake_splits):
    splits, _, _ = fake_splits
    assert read_fold_val(splits, fold=3) == ["zeta", "alpha", "mid", "omega", "extra"]


def test_read_labels_parses_identifier_and_label(fake_splits):
    _, csv_path, _ = fake_splits
    labels = read_labels(csv_path)
    assert labels["zeta"] == 1 and labels["mid"] == 0
    assert len(labels) == 5


def test_select_cases_takes_the_first_n_in_file_order(fake_splits):
    splits, csv_path, root = fake_splits
    refs = select_cases(splits, csv_path, count=4, fold=3, data_root=root)
    assert [r.case_id for r in refs] == ["zeta", "alpha", "mid", "omega"]
    assert [r.label for r in refs] == [1, 1, 0, 0]
    assert [r.label_name for r in refs] == ["Malignant", "Malignant", "Benign", "Benign"]
    assert all(isinstance(r, CaseRef) for r in refs)


def test_select_cases_raises_on_missing_file(fake_splits):
    splits, csv_path, root = fake_splits
    (root / "labelsTr" / "alpha.nii.gz").unlink()
    with pytest.raises(FileNotFoundError, match="alpha"):
        select_cases(splits, csv_path, count=4, fold=3, data_root=root)


def test_select_cases_raises_when_label_is_unknown(fake_splits):
    splits, csv_path, root = fake_splits
    csv_path.write_text("identifier,label\nzeta,1\n")
    with pytest.raises(KeyError, match="alpha"):
        select_cases(splits, csv_path, count=4, fold=3, data_root=root)


def test_select_cases_raises_when_fold_is_too_small(fake_splits):
    splits, csv_path, root = fake_splits
    with pytest.raises(ValueError, match="only 1"):
        select_cases(splits, csv_path, count=4, fold=1, data_root=root)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_jsc_cases.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'gradcam_repro.jsc.cases'`

- [ ] **Step 3: Write the implementation**

Create `src/gradcam_repro/jsc/cases.py`:

```python
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

FOLD = 3
LABEL_NAMES = {0: "Benign", 1: "Malignant"}


@dataclass(frozen=True)
class CaseRef:
    case_id: str
    label: int
    label_name: str
    image_path: Path
    label_path: Path


def read_labels(cls_data_csv: Path) -> dict[str, int]:
    with Path(cls_data_csv).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{cls_data_csv} has no rows")
    for column in ("identifier", "label"):
        if column not in rows[0]:
            raise KeyError(f"{cls_data_csv} is missing the {column!r} column")
    return {row["identifier"]: int(row["label"]) for row in rows}


def read_fold_val(splits_json: Path, fold: int = FOLD) -> list[str]:
    """Validation identifiers for one fold, in the order the file lists them.

    The published splits are a custom stratified division, not nnU-Net's default
    KFold: the five val sets hold 1118 / 1285 / 1214 / 1289 / 1226 cases.
    """
    splits = json.loads(Path(splits_json).read_text())
    if not isinstance(splits, list) or fold >= len(splits):
        raise ValueError(f"{splits_json} has no fold {fold}")
    return list(splits[fold]["val"])


def select_cases(
    splits_json: Path,
    cls_data_csv: Path,
    count: int = 4,
    fold: int = FOLD,
    data_root: Path = Path("data/luna25"),
) -> list[CaseRef]:
    """The first `count` validation cases of `fold`, in file order.

    Deliberately unsorted and unfiltered — taking the head of the model's own
    held-out list keeps the choice reproducible and free of cherry-picking.
    Held-out matters: cases from the train split would have been memorised.
    """
    val = read_fold_val(splits_json, fold)
    if len(val) < count:
        raise ValueError(f"fold {fold} has only {len(val)} validation cases, need {count}")
    labels = read_labels(cls_data_csv)
    root = Path(data_root)

    refs: list[CaseRef] = []
    for case_id in val[:count]:
        if case_id not in labels:
            raise KeyError(f"no classification label for {case_id}")
        image_path = root / "imagesTr" / f"{case_id}_0000.nii.gz"
        label_path = root / "labelsTr" / f"{case_id}.nii.gz"
        for path in (image_path, label_path):
            if not path.is_file():
                raise FileNotFoundError(f"missing {path} for case {case_id}")
        label = labels[case_id]
        refs.append(
            CaseRef(
                case_id=case_id,
                label=label,
                label_name=LABEL_NAMES[label],
                image_path=image_path,
                label_path=label_path,
            )
        )
    return refs
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_jsc_cases.py -v`
Expected: 6 passed.

- [ ] **Step 5: Verify against the real splits (manual, needs gated metadata)**

Run:

```bash
.venv/bin/python -c "
from pathlib import Path
from gradcam_repro.jsc.cases import select_cases
refs = select_cases(Path('artifacts/jsc/luna25_meta/splits_final.json'),
                    Path('artifacts/jsc/luna25_meta/cls_data.csv'))
for r in refs: print(f'{r.case_id:22s} {r.label_name}')
assert [r.case_id for r in refs] == ['100012_1_19990102','100012_1_20000102','100289_4_20010102','100438_1_19990102']
assert [r.label for r in refs] == [1,1,0,0]
print('SELECTION MATCHES THE SPEC')
"
```

Expected: the four spec case ids in order, then `SELECTION MATCHES THE SPEC`.

- [ ] **Step 6: Commit**

```bash
git add src/gradcam_repro/jsc/cases.py tests/test_jsc_cases.py
git commit -m "feat(jsc): reproducible fold-3 validation case selection"
```

---

### Task 6: `enrichment` and `cam_seg_agreement` metrics

Raw `mass_in_gt` stops working here. The real nodule masks cover 0.023%–0.498%
of the volume against the retired toy ground truth's 1.05%, so the number
collapses toward zero **and stops being comparable between cases** — the 0.023%
case cannot reach the 0.498% case's mass at equal attribution quality.

**Files:**
- Modify: `src/gradcam_repro/evaluate.py:14-26` (`score_single_sample`), `:29-53` (`score_attributions`)
- Modify: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `jsc.adapter.JscClsAdapter`.
- Produces:
  - `score_single_sample(heatmap, mask, predicted_mask=None) -> dict[str, float]` with keys `mass_in_gt`, `enrichment`, `inside_outside_ratio`, `pointing_acc`, and `cam_seg_agreement` when `predicted_mask` is given
  - `METRIC_KEYS: tuple[str, ...]` — the headline order: `("enrichment", "inside_outside_ratio", "pointing_acc", "cam_seg_agreement", "mass_in_gt")`
  - `soft_iou(heatmap, mask) -> float`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evaluate.py`:

```python
import pytest
import torch

from gradcam_repro.evaluate import METRIC_KEYS, score_single_sample, soft_iou


def _mask(size: int = 16, half: int = 2) -> torch.Tensor:
    m = torch.zeros(1, 1, size, size, size)
    c = size // 2
    m[:, :, c - half : c + half, c - half : c + half, c - half : c + half] = 1.0
    return m


def test_enrichment_of_a_uniform_heatmap_is_one():
    mask = _mask()
    uniform = torch.ones_like(mask)
    assert score_single_sample(uniform, mask)["enrichment"] == pytest.approx(1.0, abs=1e-4)


def test_enrichment_exceeds_one_when_heat_sits_on_the_lesion():
    mask = _mask()
    focused = mask.clone()
    assert score_single_sample(focused, mask)["enrichment"] > 10.0


def test_enrichment_below_one_when_heat_avoids_the_lesion():
    mask = _mask()
    avoiding = 1.0 - mask
    assert score_single_sample(avoiding, mask)["enrichment"] < 1.0


def test_metric_keys_lead_with_enrichment_not_raw_mass():
    assert METRIC_KEYS[0] == "enrichment"
    assert METRIC_KEYS[-1] == "mass_in_gt"


def test_soft_iou_is_one_for_identical_binary_maps():
    mask = _mask()
    assert soft_iou(mask, mask) == pytest.approx(1.0, abs=1e-6)


def test_soft_iou_is_zero_for_disjoint_maps():
    mask = _mask()
    assert soft_iou(1.0 - mask, mask) == pytest.approx(0.0, abs=1e-6)


def test_cam_seg_agreement_only_appears_when_a_prediction_is_supplied():
    mask = _mask()
    heat = mask.clone()
    assert "cam_seg_agreement" not in score_single_sample(heat, mask)
    scored = score_single_sample(heat, mask, predicted_mask=mask)
    assert scored["cam_seg_agreement"] == pytest.approx(1.0, abs=1e-6)


def test_zero_volume_mask_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        score_single_sample(torch.ones(1, 1, 4, 4, 4), torch.zeros(1, 1, 4, 4, 4))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: FAIL, `ImportError: cannot import name 'METRIC_KEYS'`

- [ ] **Step 3: Write the implementation**

Replace `score_single_sample` in `src/gradcam_repro/evaluate.py` and add
`soft_iou` and `METRIC_KEYS`:

```python
METRIC_KEYS: tuple[str, ...] = (
    "enrichment",
    "inside_outside_ratio",
    "pointing_acc",
    "cam_seg_agreement",
    "mass_in_gt",
)


def soft_iou(heatmap: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> float:
    """Soft intersection-over-union between a [0,1] heatmap and a binary mask."""
    intersection = torch.minimum(heatmap, mask).sum()
    union = torch.maximum(heatmap, mask).sum()
    return float((intersection / union.clamp_min(eps)).item())


def score_single_sample(
    heatmap: torch.Tensor,
    mask: torch.Tensor,
    predicted_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    """Attribution-quality metrics for one heatmap against one lesion mask.

    `enrichment` is the headline number, not `mass_in_gt`. LUNA25 nodules occupy
    0.023%-0.498% of a patch, so raw mass collapses toward zero and is not
    comparable across cases. Dividing by the mask's volume fraction fixes both:
    a uniform heatmap scores exactly 1.0 whatever the lesion size.
    """
    mask_voxels = mask.sum()
    if float(mask_voxels.item()) <= 0:
        raise ValueError("cannot score against an empty mask")

    inv_mask = 1.0 - mask
    mass = heatmap.sum().clamp_min(1e-8)
    inside_mass = (heatmap * mask).sum()
    inside_mean = inside_mass / mask_voxels.clamp_min(1)
    outside_mean = (heatmap * inv_mask).sum() / inv_mask.sum().clamp_min(1)
    peak_idx = int(heatmap.flatten().argmax().item())
    mass_fraction = float((inside_mass / mass).item())
    volume_fraction = float((mask_voxels / mask.numel()).item())

    scores = {
        "enrichment": mass_fraction / max(volume_fraction, 1e-12),
        "inside_outside_ratio": float((inside_mean / outside_mean.clamp_min(1e-8)).item()),
        "pointing_acc": float(mask.flatten()[peak_idx].item() > 0),
        "mass_in_gt": mass_fraction,
    }
    if predicted_mask is not None:
        scores["cam_seg_agreement"] = soft_iou(heatmap, predicted_mask)
    return scores
```

Then update `score_attributions` so its accumulator is built from the keys the
first scored sample actually returns, rather than a hardcoded triple:

```python
def score_attributions(
    model: nn.Module,
    cases: list,
    device: torch.device,
    methods: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Average each method's metrics over prepared cases.

    `cases` are `jsc.preprocess.PreparedCase` instances. The model's own
    predicted nodule mask is computed once per case and reused for every
    method's `cam_seg_agreement`.
    """
    methods = methods or DEFAULT_METHODS
    totals: dict[str, dict[str, float]] = {method: {} for method in methods}
    for case in cases:
        image = case.image.to(device)
        mask = case.mask.to(device)
        predicted = model.predict_seg(image) if hasattr(model, "predict_seg") else None
        for method in methods:
            heatmap = METHODS[method](model, image, 1)
            per = score_single_sample(heatmap, mask, predicted_mask=predicted)
            for key, value in per.items():
                totals[method][key] = totals[method].get(key, 0.0) + value
    n = max(len(cases), 1)
    return {
        method: {metric: value / n for metric, value in metrics.items()}
        for method, metrics in totals.items()
    }
```

Update the imports at the top of `evaluate.py`: drop `from .data import
CrossHalfDataset` and `from .model import ToyCNN`, add `from torch import nn`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_evaluate.py -v`
Expected: all pass, including the 8 new tests.

- [ ] **Step 5: Commit**

```bash
git add src/gradcam_repro/evaluate.py tests/test_evaluate.py
git commit -m "feat(evaluate): enrichment metric and CAM/segmentation agreement"
```

---

### Task 7: `jsc-attribute` CLI command and the first real figure

This is the review gate. Stop here and show the user one Grad-CAM figure before
building the remaining six method renderers — slice choice and colour mapping are
cheaper to correct now than after everything is wired.

**Files:**
- Create: `src/gradcam_repro/jsc/pipeline.py`
- Modify: `src/gradcam_repro/cli.py` (add `command_jsc_attribute` and its subparser)
- Create: `tests/test_jsc_pipeline.py`

**Interfaces:**
- Consumes: `loader.load_jsc_network`, `adapter.JscClsAdapter`, `cases.select_cases`, `preprocess.load_case`, `evaluate.score_single_sample`.
- Produces:
  - `@dataclass(frozen=True) class JscRunConfig` — `checkpoint: Path`, `splits: Path`, `cls_data: Path`, `data_root: Path`, `output: Path`, `methods: tuple[str, ...]`, `device: str`, `count: int`, `occlusion_mask: int`, `occlusion_stride: int`, `ig_steps: int`
  - `run_attribution(config: JscRunConfig) -> dict` — returns `{"cases": [...], "scores": {...}}` and writes `<output>/scores.json`
  - `DEFAULT_JSC_METHODS: tuple[str, ...]` — the seven display methods in deck order
- CLI: `uv run gradcam-repro jsc-attribute --checkpoint ... --methods gradcam --out artifacts/jsc/run`

- [ ] **Step 1: Write the failing test**

Create `tests/test_jsc_pipeline.py`:

```python
from __future__ import annotations

import pytest

from gradcam_repro.jsc.pipeline import DEFAULT_JSC_METHODS, JscRunConfig


def test_default_methods_are_the_seven_deck_methods():
    assert DEFAULT_JSC_METHODS == (
        "notgradcam",
        "gradcam",
        "guided_gradcam",
        "layercam",
        "occlusion",
        "integrated_gradients",
        "integrated_gradcam",
    )


def test_config_defaults_match_the_spec(tmp_path):
    config = JscRunConfig(
        checkpoint=tmp_path / "ck.pth",
        splits=tmp_path / "s.json",
        cls_data=tmp_path / "c.csv",
        data_root=tmp_path,
        output=tmp_path / "out",
    )
    assert config.count == 4
    assert config.device == "mps"
    assert config.occlusion_mask == 16
    assert config.occlusion_stride == 8
    assert config.ig_steps == 16
    assert config.methods == DEFAULT_JSC_METHODS


def test_run_attribution_rejects_a_missing_checkpoint(tmp_path):
    from gradcam_repro.jsc.pipeline import run_attribution

    config = JscRunConfig(
        checkpoint=tmp_path / "absent.pth",
        splits=tmp_path / "s.json",
        cls_data=tmp_path / "c.csv",
        data_root=tmp_path,
        output=tmp_path / "out",
    )
    with pytest.raises(FileNotFoundError):
        run_attribution(config)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_jsc_pipeline.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'gradcam_repro.jsc.pipeline'`

- [ ] **Step 3: Write the implementation**

Create `src/gradcam_repro/jsc/pipeline.py`:

```python
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from ..attribution import METHODS
from ..evaluate import score_single_sample
from .adapter import JscClsAdapter
from .cases import select_cases
from .loader import load_jsc_network, resolve_device
from .preprocess import load_case

logger = logging.getLogger(__name__)

DEFAULT_JSC_METHODS: tuple[str, ...] = (
    "notgradcam",
    "gradcam",
    "guided_gradcam",
    "layercam",
    "occlusion",
    "integrated_gradients",
    "integrated_gradcam",
)


@dataclass(frozen=True)
class JscRunConfig:
    checkpoint: Path
    splits: Path
    cls_data: Path
    data_root: Path
    output: Path
    methods: tuple[str, ...] = DEFAULT_JSC_METHODS
    device: str = "mps"
    count: int = 4
    occlusion_mask: int = 16
    occlusion_stride: int = 8
    ig_steps: int = 16


def _method_kwargs(config: JscRunConfig, method: str) -> dict:
    if method == "occlusion":
        return {
            "mask_size": config.occlusion_mask,
            "stride": config.occlusion_stride,
            "batch_size": 1,
        }
    if method in ("integrated_gradients", "integrated_gradcam"):
        return {"steps": config.ig_steps}
    return {}


def run_attribution(config: JscRunConfig) -> dict:
    """Run every configured attribution method over the selected cases."""
    for name, path in (("checkpoint", config.checkpoint), ("splits", config.splits),
                       ("cls_data", config.cls_data)):
        if not Path(path).is_file():
            raise FileNotFoundError(f"{name} not found: {path}")
    unknown = [m for m in config.methods if m not in METHODS]
    if unknown:
        raise KeyError(f"unknown attribution methods: {unknown}")

    network, checkpoint = load_jsc_network(config.checkpoint, config.device)
    model = JscClsAdapter(network)
    device = resolve_device(config.device)
    plans = checkpoint["init_args"]["plans"]

    refs = select_cases(config.splits, config.cls_data, config.count,
                        data_root=config.data_root)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)

    case_records: list[dict] = []
    for ref in refs:
        prepared = load_case(ref.image_path, ref.label_path, plans)
        image = prepared.image.to(device)
        mask = prepared.mask.to(device)
        with torch.no_grad():
            logit = float(model(image)[0, 0].item())
        predicted = model.predict_seg(image)
        record = {
            "case_id": ref.case_id,
            "label": ref.label,
            "label_name": ref.label_name,
            "logit": logit,
            "probability": float(torch.sigmoid(torch.tensor(logit)).item()),
            "original_shape": list(prepared.original_shape),
            "original_spacing": list(prepared.original_spacing),
            "resampled_shape": list(prepared.resampled_shape),
            "transform": prepared.transform,
            "gt_voxels": int(mask.sum().item()),
            "predicted_voxels": int(predicted.sum().item()),
            "methods": {},
        }
        for method in config.methods:
            started = time.perf_counter()
            heatmap = METHODS[method](model, image, 1, **_method_kwargs(config, method))
            elapsed = time.perf_counter() - started
            if not torch.isfinite(heatmap).all():
                raise RuntimeError(f"{method} produced non-finite values for {ref.case_id}")
            if float(heatmap.max().item()) <= 0:
                raise RuntimeError(f"{method} produced an all-zero map for {ref.case_id}")
            torch.save(heatmap.detach().cpu(), output / f"{ref.case_id}__{method}.pt")
            record["methods"][method] = {
                "seconds": round(elapsed, 3),
                **score_single_sample(heatmap, mask, predicted_mask=predicted),
            }
            logger.info("%s / %s: %.1fs", ref.case_id, method, elapsed)
        torch.save(
            {"image": prepared.image, "mask": prepared.mask, "predicted": predicted.cpu()},
            output / f"{ref.case_id}__inputs.pt",
        )
        case_records.append(record)

    scores = {
        method: {
            key: sum(c["methods"][method][key] for c in case_records) / len(case_records)
            for key in case_records[0]["methods"][method]
        }
        for method in config.methods
    }
    result = {"cases": case_records, "scores": scores}
    (output / "scores.json").write_text(json.dumps(result, indent=2))
    return result
```

Add to `src/gradcam_repro/cli.py`, after `command_web_export`:

```python
def command_jsc_attribute(args: argparse.Namespace) -> None:
    from .jsc.pipeline import DEFAULT_JSC_METHODS, JscRunConfig, run_attribution

    config = JscRunConfig(
        checkpoint=Path(args.checkpoint),
        splits=Path(args.splits),
        cls_data=Path(args.cls_data),
        data_root=Path(args.data_root),
        output=Path(args.out),
        methods=tuple(args.methods) if args.methods else DEFAULT_JSC_METHODS,
        device=args.device,
        count=args.count,
        occlusion_mask=args.occlusion_mask,
        occlusion_stride=args.occlusion_stride,
        ig_steps=args.ig_steps,
    )
    result = run_attribution(config)
    print(json.dumps(result["scores"], indent=2))
```

and register the subparser next to the others:

```python
    jsc_parser = subparsers.add_parser(
        "jsc-attribute",
        help="Run the seven attribution methods against the JSC LUNA25 checkpoint.",
    )
    jsc_parser.add_argument("--checkpoint", default="artifacts/jsc/fold_3/checkpoint_best.pth")
    jsc_parser.add_argument("--splits", default="artifacts/jsc/luna25_meta/splits_final.json")
    jsc_parser.add_argument("--cls-data", dest="cls_data", default="artifacts/jsc/luna25_meta/cls_data.csv")
    jsc_parser.add_argument("--data-root", dest="data_root", default="data/luna25")
    jsc_parser.add_argument("--out", default="artifacts/jsc/run")
    jsc_parser.add_argument("--methods", nargs="*", default=None)
    jsc_parser.add_argument("--device", default="mps")
    jsc_parser.add_argument("--count", type=int, default=4)
    jsc_parser.add_argument("--occlusion-mask", dest="occlusion_mask", type=int, default=16)
    jsc_parser.add_argument("--occlusion-stride", dest="occlusion_stride", type=int, default=8)
    jsc_parser.add_argument("--ig-steps", dest="ig_steps", type=int, default=16)
    jsc_parser.set_defaults(func=command_jsc_attribute)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_jsc_pipeline.py -v`
Expected: 3 passed.

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 5: Run Grad-CAM only on one case (fast — skips occlusion)**

Run:

```bash
cd /Users/baiduli/ProgramProject/gradcam-repro
uv run gradcam-repro jsc-attribute --methods gradcam --count 1 --out artifacts/jsc/run-smoke
```

Expected: a JSON block with `gradcam` metrics; `enrichment` should be
noticeably above 1.0 for `100012_1_19990102` (Malignant, 760-voxel nodule). If
`enrichment` lands near 1.0, stop and investigate before continuing — either
preprocessing or the tap point is wrong.

- [ ] **Step 6: Render one figure and show it to the user**

Run:

```bash
.venv/bin/python -c "
import torch, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
out = Path('artifacts/jsc/run-smoke')
cid = '100012_1_19990102'
inputs = torch.load(out / f'{cid}__inputs.pt', weights_only=True)
heat = torch.load(out / f'{cid}__gradcam.pt', weights_only=True)
ct, gt, pred = inputs['image'][0,0], inputs['mask'][0,0], inputs['predicted'][0,0]
z = int(gt.sum(dim=(1,2)).argmax())           # slice with the most nodule
fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
axes[0].imshow(ct[z], cmap='gray'); axes[0].set_title(f'CT z={z}')
axes[1].imshow(ct[z], cmap='gray'); axes[1].contour(gt[z], levels=[0.5], colors='lime'); axes[1].set_title('GT nodule')
axes[2].imshow(ct[z], cmap='gray'); axes[2].contour(pred[z], levels=[0.5], colors='cyan'); axes[2].set_title('predicted nodule')
axes[3].imshow(ct[z], cmap='gray'); axes[3].imshow(heat[0,0,z], cmap='turbo', alpha=0.5); axes[3].set_title('Grad-CAM')
for a in axes: a.axis('off')
fig.suptitle(f'{cid}  Malignant')
fig.tight_layout(); fig.savefig(out / 'smoke_gradcam.png', dpi=140)
print('wrote', out / 'smoke_gradcam.png')
"
```

Show `artifacts/jsc/run-smoke/smoke_gradcam.png` to the user and confirm the
slice choice, colour map, and overlay opacity before proceeding to Task 8.

- [ ] **Step 7: Commit**

```bash
git add src/gradcam_repro/jsc/pipeline.py src/gradcam_repro/cli.py tests/test_jsc_pipeline.py
git commit -m "feat(cli): jsc-attribute command running all seven methods"
```

---

### Task 8: Figures — method grid, Benign/Malignant pairing, longitudinal panel

**Files:**
- Create: `src/gradcam_repro/jsc/figures.py`
- Modify: `src/gradcam_repro/visualize.py` (reuse `overlay_heatmap`, `volume_slice`, `normalize_panel`; do not duplicate them)
- Create: `tests/test_jsc_figures.py`

**Interfaces:**
- Consumes: `pipeline.run_attribution` output layout under `<output>/`.
- Produces:
  - `nodule_slice(mask: torch.Tensor) -> int` — the z index holding the most nodule
  - `render_method_grid(run_dir: Path, case_id: str, methods, out: Path) -> Path` — rows: CT / GT / predicted / seven heatmaps
  - `render_class_pairing(run_dir: Path, malignant_ids, benign_ids, methods, out: Path) -> Path` — replaces the retired `class_discriminability.png`
  - `render_longitudinal(run_dir: Path, earlier_id: str, later_id: str, methods, out: Path) -> Path`
  - `FIGURE_CAPTIONS: dict[str, str]` — the declared-deviation captions

`FIGURE_CAPTIONS` must state, verbatim: guided backprop applies the ReLU rule to
LeakyReLU; the occluder is 16 voxels cubed, physically 32 mm x 10.4 mm x 10.4 mm
and therefore elongated in z; `100438_1_19990102` was centre-cropped from
(64,135,135); the two `100012_1` scans are the same patient and not independent
samples.

- [ ] **Step 1: Write the failing test**

Create `tests/test_jsc_figures.py`:

```python
from __future__ import annotations

import torch

from gradcam_repro.jsc.figures import FIGURE_CAPTIONS, nodule_slice, render_method_grid


def _fake_run(tmp_path, case_id="c0", methods=("gradcam", "layercam")):
    mask = torch.zeros(1, 1, 8, 16, 16)
    mask[:, :, 5, 6:10, 6:10] = 1.0            # nodule concentrated on z=5
    torch.save(
        {"image": torch.rand(1, 1, 8, 16, 16), "mask": mask, "predicted": mask.clone()},
        tmp_path / f"{case_id}__inputs.pt",
    )
    for method in methods:
        torch.save(torch.rand(1, 1, 8, 16, 16), tmp_path / f"{case_id}__{method}.pt")
    return tmp_path


def test_nodule_slice_picks_the_densest_slice():
    mask = torch.zeros(1, 1, 8, 16, 16)
    mask[:, :, 5, 6:10, 6:10] = 1.0
    mask[:, :, 2, 6:8, 6:8] = 1.0
    assert nodule_slice(mask) == 5


def test_render_method_grid_writes_a_png(tmp_path):
    run_dir = _fake_run(tmp_path / "run")
    out = tmp_path / "grid.png"
    result = render_method_grid(run_dir, "c0", ("gradcam", "layercam"), out)
    assert result == out
    assert out.is_file()
    assert out.stat().st_size > 1000


def test_captions_declare_every_deviation():
    text = " ".join(FIGURE_CAPTIONS.values()).lower()
    for fragment in ("leakyrelu", "32 mm", "centre-crop", "not independent"):
        assert fragment in text, fragment
```

Note: `_fake_run` must `mkdir` the run directory. Add
`(tmp_path / "run").mkdir(parents=True, exist_ok=True)` before saving, or pass
an existing directory.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_jsc_figures.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'gradcam_repro.jsc.figures'`

- [ ] **Step 3: Write the implementation**

Create `src/gradcam_repro/jsc/figures.py` implementing exactly the interfaces
above. Requirements:

- `nodule_slice(mask)` returns `int(mask[0, 0].sum(dim=(1, 2)).argmax())`.
- Every renderer loads `<run_dir>/<case_id>__inputs.pt` and
  `<run_dir>/<case_id>__<method>.pt` with `weights_only=True`, raising
  `FileNotFoundError` naming the missing file.
- Axial slices are drawn without aspect correction: the y-x plane is isotropic
  at 0.664 mm. Do not pass `aspect=` to `imshow`.
- Heatmaps use `cmap="turbo"`, `alpha=0.5`, over a grayscale CT.
- GT contours are `lime`, predicted contours are `cyan`, both at `levels=[0.5]`.
- `matplotlib.use("Agg")` at import, so tests never need a display.
- `FIGURE_CAPTIONS` keys: `"guided_gradcam"`, `"occlusion"`, `"crop"`,
  `"longitudinal"`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_jsc_figures.py -v`
Expected: 3 passed.

- [ ] **Step 5: Render the full deck figures (~72 min — occlusion dominates)**

Run:

```bash
cd /Users/baiduli/ProgramProject/gradcam-repro
uv run gradcam-repro jsc-attribute --out artifacts/jsc/run
```

Expected: ~18 min per case, four cases. Watch that `enrichment` for `occlusion`
is above 1.0 — an all-zero occlusion map would have raised in Task 7's guard,
but a near-1.0 enrichment means the occluder is washing out the signal.

- [ ] **Step 6: Commit**

```bash
git add src/gradcam_repro/jsc/figures.py tests/test_jsc_figures.py
git commit -m "feat(jsc): method grid, class pairing, and longitudinal figures"
```

---

### Task 9: Web bundle v2

**Files:**
- Modify: `src/gradcam_repro/web_export.py` (`WEB_BUNDLE_SCHEMA:20`, `build_model_graph:33`, `_METRIC_KEYS:200`, `build_benchmark:203`, `export_example:150`, `web_export:261`)
- Modify: `web/DATA_CONTRACT.md`
- Modify: `tests/test_web_export_graph.py`, `tests/test_web_export_benchmark.py`, `tests/test_web_export_manifest.py`, `tests/test_web_export_slice.py`, `tests/test_web_export_activations.py`, `tests/test_web_export_example.py`, `tests/test_web_export_integration.py`

**Interfaces:**
- Consumes: `evaluate.METRIC_KEYS`, `pipeline.run_attribution` output, `adapter.JscClsAdapter`.
- Produces: `WEB_BUNDLE_SCHEMA = "gradcam-repro.web-bundle.v2"`; `build_model_graph(model: JscClsAdapter, input_shape) -> dict` describing the 6-stage encoder plus FPN, conv_block, GAP and classifier; `_METRIC_KEYS` becomes `evaluate.METRIC_KEYS`; per-example `meta.json` gains `label`, `label_name`, `logit`, `probability`, `resampled_shape`, `transform`, `gt_voxels`, `predicted_voxels`; each example directory gains `predicted_mask.png`.

- [ ] **Step 1: Update the contract document first**

Edit `web/DATA_CONTRACT.md`: change every `gradcam-repro.web-bundle.v1` to
`...v2`, replace the frozen-contract note with a line recording that v1 described
the retired MSD toy backend, document the new `meta.json` fields and
`predicted_mask.png`, and replace the `_METRIC_KEYS` triple with the five keys
from `evaluate.METRIC_KEYS`.

- [ ] **Step 2: Update the seven tests to expect v2**

Run: `.venv/bin/python -m pytest tests/ -q -k web_export`
Expected: FAIL, schema-string and metric-key assertions.

Change each `"gradcam-repro.web-bundle.v1"` literal to `"...v2"`, and in
`tests/test_web_export_benchmark.py` assert against `evaluate.METRIC_KEYS`
rather than the hardcoded triple. Replace the `real_ct_model` fixture usage with
`mini_jsc_net` wrapped in `JscClsAdapter`.

- [ ] **Step 3: Update `web_export.py`**

Bump the schema constant, retarget `build_model_graph` at the JSC adapter
(`model.network.seg_network.encoder.stages`, then FPN / conv_block / gap /
classifier as nodes), import `METRIC_KEYS` from `evaluate` instead of the local
triple, write `predicted_mask.png` in `export_example` using
`_save_mask_png(predicted_slice, ct_slice, path)`, and extend the per-example
`meta.json` with the fields listed above.

- [ ] **Step 4: Run the web-export tests**

Run: `.venv/bin/python -m pytest tests/ -q -k web_export`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/gradcam_repro/web_export.py web/DATA_CONTRACT.md tests/test_web_export_*.py
git commit -m "feat(web): bundle schema v2 for the JSC LUNA25 backend"
```

---

### Task 10: Retire the MSD path and refresh the docs

**Files:**
- Delete: `src/gradcam_repro/real_ct.py`
- Modify: `src/gradcam_repro/model.py` (remove `RealCtCNN`, keep `ToyCNN`)
- Modify: `src/gradcam_repro/cli.py` (remove `command_real_preprocess`, `command_real_summary`, `command_real_train`, `command_real_demo` and their four subparsers)
- Modify: `src/gradcam_repro/visualize.py` (drop the `RealCtCNN` import and any MSD-only renderer)
- Modify: `tests/conftest.py` (remove the `real_ct_model`, `fake_sample`, `fake_cache` fixtures once nothing uses them)
- Delete: `tests/test_conftest_smoke.py` if it only exercises retired fixtures
- Modify: `README.md`
- Modify: `scripts/render_real_ct_deck_figures.py` → rename to `scripts/render_jsc_deck_figures.py`

- [ ] **Step 1: Find every remaining reference**

Run:

```bash
cd /Users/baiduli/ProgramProject/gradcam-repro
grep -rn 'real_ct\|RealCtCNN\|RealCt\|msd\|Task06' src/ tests/ scripts/ web/ README.md --include='*.py' --include='*.md' | grep -v '^docs/'
```

Every hit must be either removed or repointed at the JSC path. Record the list
before editing so nothing is missed.

- [ ] **Step 2: Remove the code**

Delete `src/gradcam_repro/real_ct.py`. Strip `RealCtCNN` from `model.py`. Remove
the four `real-*` CLI commands and subparsers. Keep `train` / `demo` / `score` /
`deck` / `all` working — they use `ToyCNN` and `CrossHalfDataset`, which stay.

- [ ] **Step 3: Run the whole suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass, no import errors.

Run: `.venv/bin/python -c "from gradcam_repro.cli import main; main()" --help`
Expected: help text listing `train`, `demo`, `score`, `deck`, `all`,
`web-export`, `jsc-attribute` — and no `real-*` command.

- [ ] **Step 4: Rewrite the README sections**

Replace the "Reproduce Current Deck" and "Real CT Example" sections with the JSC
workflow. The new reproduce path is:

```bash
uv pip install -e third_party/JSC && uv pip install torchmetrics
# fetch weights + the four cases: see docs/jsc-luna25-sources.md
uv run gradcam-repro jsc-attribute --out artifacts/jsc/run
uv run gradcam-repro web-export --run artifacts/jsc/run --out web/public/data
```

State the model (JSC `nnUNetCLSTrainerMTL__nnUNetPlans__3d_fullres`, fold 3,
PlainConvUNet backbone), the data (LUNA25 malignancy classification), the CAM tap
point (`conv_block`, `(1,640,16,16,16)`), and link both design documents. Correct
the stale note claiming Apple MPS lacks a 3D-convolution kernel: measured working
and 1.9x faster than CPU on this machine, numerically identical to 4.8e-07.

- [ ] **Step 5: Full verification**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass.

Run: `uv run gradcam-repro demo` then `uv run gradcam-repro score`
Expected: the legacy synthetic path still produces figures and scores.

- [ ] **Step 6: Commit**

```bash
git add -A src/ tests/ scripts/ README.md
git commit -m "refactor: retire the MSD Task06_Lung backend"
```

---

## Self-review notes

**Spec coverage.** M1 → Tasks 1–2; M2 → Task 4; M3 → Task 3 (plus the adapter in
Task 2, which is what lets `attribution.py` survive as a patch rather than a
rewrite); M4 → Task 6; M5 → Tasks 8–9. The five declared deviations land in:
centre-crop → Task 4 `fit_to_patch` and Task 8 `FIGURE_CAPTIONS`; decoder skip →
Task 2 adapter; LeakyReLU guided rule → Task 3 and Task 8 captions; MPS → Task 1
`resolve_device` and Task 10 README correction; no AUROC reproduction → out of
scope, unchanged. Verification items 1–7 map to Task 1 Steps 4–5, Task 4 Step 5,
Task 6 Step 4 and Task 7 Step 5.

**Two spec items deliberately left out.** The physically-isotropic 8x24x24
occluder alternative stays a documented option, not an implementation — the
default is voxel-space 16 cubed with the anisotropy stated in the caption. The
`artifacts/jsc/case_index.csv` join is already built and verified; no task
regenerates it, and `select_cases` does not depend on it.

**Naming consistency.** `CAM_LAYER` (`jsc/adapter.py`) and `TARGET_LAYER`
(`attribution.py`) both hold `"cam"`; Task 3 changes the latter and Task 2
defines the former. `PATCH_SIZE` lives only in `jsc/loader.py`, imported by
`preprocess.py`. `METRIC_KEYS` is defined in `evaluate.py` and consumed by
`web_export.py` in Task 9 — the old private `_METRIC_KEYS` is removed, not
shadowed.

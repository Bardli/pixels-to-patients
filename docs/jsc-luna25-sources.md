# JSC + LUNA25 — external sources and local artifacts

Everything the JSC / LUNA25 attribution pipeline needs that is **not** in this
repository, with the exact commands to re-fetch it and SHA-256 digests to verify
the result. Design rationale lives in
[`docs/superpowers/specs/2026-07-28-jsc-luna25-attribution-design.md`](superpowers/specs/2026-07-28-jsc-luna25-attribution-design.md).

Re-downloadable material is deliberately kept out of git: model weights
(443.7 MB) and imaging data are reproducible from the commands below, so the
repository stores only the manifest. `.gitignore` already covers `artifacts/`,
`/data/`, `*.pth` and `third_party/`.

All digests were recorded on 2026-07-28.

## 1. JSC source (nnU-Net fork)

```bash
git clone --depth 1 https://github.com/bowang-lab/JSC.git third_party/JSC
git -C third_party/JSC checkout 49511ef01c414014afb7e7a3265d820544bf93cc
```

Pinned commit `49511ef01c414014afb7e7a3265d820544bf93cc` (2026-03-27,
*"Update preprocess skill to check for existing cls_data.csv in raw folder"*).
Apache-2.0. The package declares itself as `nnunetv2` version 2.6.2, i.e. a fork
of upstream nnU-Net v2.6.2.

### Two patches are required before it installs

**(a) setuptools cannot build it as shipped.** The repo root holds both
`configs/` and `nnunetv2/`, so flat-layout auto-discovery aborts with
`error: Multiple top-level packages discovered in a flat-layout: ['configs', 'nnunetv2']`.
Add to `third_party/JSC/pyproject.toml`, before `[build-system]`:

```toml
[tool.setuptools.packages.find]
include = ["nnunetv2*"]
```

**(b) `torchmetrics` is missing from the dependency list** but imported at
`nnunetv2/training/nnUNetTrainer/nnUNetCLSTrainer.py:73`. Install it separately.

```bash
uv pip install -e third_party/JSC
uv pip install torchmetrics          # 1.9.0 at time of writing
```

`uv pip install` is used deliberately so `pyproject.toml` and `uv.lock` stay
untouched. **Caveat: `uv sync` prunes anything not in the lockfile — re-run both
installs after any sync.**

### Path configuration

JSC's README says to edit `nnunetv2/paths.py`. That is out of date: the fork
reads the `nnUNet_raw` / `nnUNet_preprocessed` / `nnUNet_results` environment
variables and only prints a warning when they are unset. Set the env vars.

## 2. Trained weights — ungated

Source: [`cyyu96/AutoMSC-Baselines`](https://huggingface.co/cyyu96/AutoMSC-Baselines),
path `Dataset005_LUNA25/nnUNetCLSTrainerMTL__nnUNetPlans__3d_fullres/`.
No authentication needed.

```bash
B=https://huggingface.co/cyyu96/AutoMSC-Baselines/resolve/main/Dataset005_LUNA25/nnUNetCLSTrainerMTL__nnUNetPlans__3d_fullres
mkdir -p artifacts/jsc/fold_3
for f in plans.json dataset.json dataset_fingerprint.json; do
  curl -sSL --fail -o "artifacts/jsc/$f" "$B/$f"
done
curl -sSL --fail -o artifacts/jsc/fold_3/checkpoint_best.pth "$B/fold_3/checkpoint_best.pth"
```

| File | Bytes | SHA-256 |
|---|---|---|
| `plans.json` | 10,499 | `2565b47ae982599f0f81917cd176c5b0d2d90a038f42cb9217f841383c07b0f3` |
| `dataset.json` | 870 | `9db0b8b5f37e9d962a1bfbe99c543c69cba4f299009c244b1c5695237ce80548` |
| `dataset_fingerprint.json` | 999,480 | `10b31625b4c97e051c8b64f213b2d6a44894b1027d26f4fd871aa4ff7c0741c2` |
| `fold_3/checkpoint_best.pth` | 443,676,001 | `33ad8109aafd5f0b6d6c82a7f6911fa4167c36e44c9c4e50bed3c19aef25968f` |

Folds 0–2 and 4 exist at the same path and are not used here. Published 5-fold
means: DSC 0.738, AUROC 0.870; fold 3 specifically DSC 0.732, AUROC 0.884.

FLARE's own [`FLARE-MedFM/FLARE-AutoMSC-Val-Baseline`](https://huggingface.co/FLARE-MedFM/FLARE-AutoMSC-Val-Baseline)
ships only Dataset007/072_GliomaIDHType — the LUNA25 weights exist solely in the
author's personal repo. They are a published training baseline, not the official
validation docker baseline.

### The checkpoint is self-contained

`init_args` embeds the complete `plans` and `dataset_json`, and they match the
separately downloaded `plans.json` field for field. Loading from the checkpoint
alone is sufficient; the standalone JSONs are kept for inspection.

### Safe loading

`weights_only=True` works after allowlisting six numpy globals. Do **not** fall
back to `weights_only=False`.

```python
torch.serialization.add_safe_globals([
    numpy._core.multiarray.scalar, numpy.dtype,
    numpy.dtypes.Float32DType, numpy.dtypes.Float64DType,
    numpy.dtypes.Int64DType, numpy.dtypes.Int32DType,
])
ck = torch.load(path, map_location="cpu", weights_only=True)
```

## 3. Imaging data — gated

Source: [`FLARE-MedFM/FLARE-AutoMSC`](https://huggingface.co/datasets/FLARE-MedFM/FLARE-AutoMSC),
folder `Dataset005_LUNA25`. **`gated: "manual"`** — access must be requested on
the dataset page and granted by the organisers. Log in first; the token cannot be
handled on your behalf:

```bash
huggingface-cli login
```

The full folder is 6132 images + 6132 masks, 8.38 GB. Only four cases are
needed:

```bash
TOK=$(cat ~/.cache/huggingface/token)
B=https://huggingface.co/datasets/FLARE-MedFM/FLARE-AutoMSC/resolve/main/Dataset005_LUNA25
mkdir -p artifacts/jsc/luna25_meta data/luna25/imagesTr data/luna25/labelsTr

for f in splits_final.json cls_data.csv dataset.json; do
  curl -sSL --fail -H "Authorization: Bearer $TOK" -o "artifacts/jsc/luna25_meta/$f" "$B/$f"
done

for c in 100012_1_19990102 100012_1_20000102 100289_4_20010102 100438_1_19990102; do
  curl -sSL --fail -H "Authorization: Bearer $TOK" \
    -o "data/luna25/imagesTr/${c}_0000.nii.gz" "$B/imagesTr/${c}_0000.nii.gz"
  curl -sSL --fail -H "Authorization: Bearer $TOK" \
    -o "data/luna25/labelsTr/${c}.nii.gz" "$B/labelsTr/${c}.nii.gz"
done
```

Note `-H "Authorization: Bearer ..."` is required — `curl` does not pick up the
huggingface-cli session on its own. Always check the HTTP status: a gated miss
returns 401 with a short HTML body, which silently overwrites the target file if
you do not pass `--fail`.

| File | Bytes | SHA-256 |
|---|---|---|
| `luna25_meta/splits_final.json` | 828,397 | `cf4cdd44ad370e40e670379afaaa1fd1eed0df727625b7f4bf089bf0d8d3a802` |
| `luna25_meta/cls_data.csv` | 122,724 | `4fbaf9b32364e4c3cf01293df7c3f492b5e7ea4d337de9b244a39a3121e4e82b` |
| `luna25_meta/dataset.json` | 870 | `3a5bb526f2bf617476d72cafd5178901798e317ad445182a533153c989309668` |
| `imagesTr/100012_1_19990102_0000.nii.gz` | 1,476,862 | `901f628855eee3408052d8a1f444a75412d20080f02a0b06adc8fbdb57ff3a56` |
| `labelsTr/100012_1_19990102.nii.gz` | 1,349 | `77035402c2d31869242dfe5b98801acd9d022cbb13ba9bf5c3c99f30fc777456` |
| `imagesTr/100012_1_20000102_0000.nii.gz` | 1,456,008 | `cff8ca8048f75da3faf5a92820e225016ebccbfc5ef47993e93779e29bb16097` |
| `labelsTr/100012_1_20000102.nii.gz` | 1,897 | `2f6d6aa436b9349b02ce72d221ff849234ada897fdb63884e9ecda4209c12023` |
| `imagesTr/100289_4_20010102_0000.nii.gz` | 1,346,323 | `e398720db951fbd3f61a3114d57949b87ec50839200aa2f77c094f83a242a5c6` |
| `labelsTr/100289_4_20010102.nii.gz` | 1,241 | `4ef9fecfcfaf415ee511fdc92eb5ed58a79506c23be594a5e9be29cf3ce75f1f` |
| `imagesTr/100438_1_19990102_0000.nii.gz` | 1,400,721 | `5e33f540265691f8eb0d47c8151f451a3f0a04c6b809a1561b9ce4a3bc8fca28` |
| `labelsTr/100438_1_19990102.nii.gz` | 1,312 | `9a398648c49fb921f0026a75984736b31c696d59a5a77c86b3e8eba7c2655ed3` |

Licence: CC BY 4.0. Cite the LUNA25 challenge paper — Peeters D, Obreja B,
Antonissen N, Jacobs C. *Benchmarking of Artificial Intelligence and
Radiologists for Lung Cancer Screening in CT: The LUNA25 Challenge.* MICCAI
2025. <https://doi.org/10.5281/zenodo.15094631>

### Gotcha: the gated `dataset.json` carries the wrong dataset name

The gated `Dataset005_LUNA25/dataset.json` sets `"name": "Dataset009_LUNA25"`.
The two files are byte-identical apart from that one line — the checkpoint's copy
says `Dataset005_LUNA25`. The same 009 numbering shows up in
`cyyu96/AutoMSC_examples`, so 009 was evidently the author's working ID.

nnU-Net resolves paths through this field (`maybe_convert_to_dataset_name`), so a
`nnUNet_raw/Dataset005_LUNA25/` folder whose `dataset.json` says 009 will send
preprocessing to the wrong directory. **Use the checkpoint's embedded
`dataset_json`**, not the gated file. The gated copy is kept only for the record.

## 4. Locally derived (not downloaded)

| File | Bytes | SHA-256 | Produced by |
|---|---|---|---|
| `artifacts/jsc/case_index.csv` | 394,245 | `4fbb96f0e209b76c91e1e66da320007fd41044fd20a0af781b1c27510e40c0e8` | HF file listing (ungated) joined to `dataset_fingerprint.json` |

One row per case for all 6132 cases: `case_id`, original shape and spacing,
shape after resampling to the plans target spacing, and a `single_patch` flag.

The join relies on the fingerprint arrays being index-aligned with the sorted
case-ID list. That holds because `dataset.json` has no `dataset` key, so
nnU-Net enumerates through `get_identifiers_from_splitted_dataset_folder`, which
returns `np.unique(...)` — sorted — and the fingerprint extractor preserves that
order. Confirmed empirically: for all four downloaded cases the on-disk NIfTI
shape and spacing match the fingerprint entry exactly.

## 5. Verify a local checkout

```bash
shasum -a 256 -c <<'EOF'
2565b47ae982599f0f81917cd176c5b0d2d90a038f42cb9217f841383c07b0f3  artifacts/jsc/plans.json
9db0b8b5f37e9d962a1bfbe99c543c69cba4f299009c244b1c5695237ce80548  artifacts/jsc/dataset.json
10b31625b4c97e051c8b64f213b2d6a44894b1027d26f4fd871aa4ff7c0741c2  artifacts/jsc/dataset_fingerprint.json
33ad8109aafd5f0b6d6c82a7f6911fa4167c36e44c9c4e50bed3c19aef25968f  artifacts/jsc/fold_3/checkpoint_best.pth
cf4cdd44ad370e40e670379afaaa1fd1eed0df727625b7f4bf089bf0d8d3a802  artifacts/jsc/luna25_meta/splits_final.json
4fbaf9b32364e4c3cf01293df7c3f492b5e7ea4d337de9b244a39a3121e4e82b  artifacts/jsc/luna25_meta/cls_data.csv
3a5bb526f2bf617476d72cafd5178901798e317ad445182a533153c989309668  artifacts/jsc/luna25_meta/dataset.json
901f628855eee3408052d8a1f444a75412d20080f02a0b06adc8fbdb57ff3a56  data/luna25/imagesTr/100012_1_19990102_0000.nii.gz
77035402c2d31869242dfe5b98801acd9d022cbb13ba9bf5c3c99f30fc777456  data/luna25/labelsTr/100012_1_19990102.nii.gz
cff8ca8048f75da3faf5a92820e225016ebccbfc5ef47993e93779e29bb16097  data/luna25/imagesTr/100012_1_20000102_0000.nii.gz
2f6d6aa436b9349b02ce72d221ff849234ada897fdb63884e9ecda4209c12023  data/luna25/labelsTr/100012_1_20000102.nii.gz
e398720db951fbd3f61a3114d57949b87ec50839200aa2f77c094f83a242a5c6  data/luna25/imagesTr/100289_4_20010102_0000.nii.gz
4ef9fecfcfaf415ee511fdc92eb5ed58a79506c23be594a5e9be29cf3ce75f1f  data/luna25/labelsTr/100289_4_20010102.nii.gz
5e33f540265691f8eb0d47c8151f451a3f0a04c6b809a1561b9ce4a3bc8fca28  data/luna25/imagesTr/100438_1_19990102_0000.nii.gz
9a398648c49fb921f0026a75984736b31c696d59a5a77c86b3e8eba7c2655ed3  data/luna25/labelsTr/100438_1_19990102.nii.gz
EOF
```

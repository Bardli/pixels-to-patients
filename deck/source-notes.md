# Source Notes

Source deck:

- `/Users/baiduli/Downloads/GradCam.pptx`
- The PPTX contains 19 full-slide images and no extractable editable text.
- Extracted slide screenshots live in `references/slides/`.
- OCR text lives in `references/ocr/`.

Rebuild approach:

- Treat the original deck as source/reference, not as an editable template.
- Recreate the content with native editable PowerPoint objects through
  artifact-tool presentation generation.
- Keep experiment figures reproducible from the local Python code.
- Preserve the original paper/source trail. Method pages now carry the citation
  line from the source PPT/OCR, with the references slide keeping the full
  chronological list.
- Keep the lecture logic as a progression of pain points: activation-only
  baseline, class-aware localization, voxel detail, local gradients,
  perturbation-based causal testing, path-integrated input attribution, and
  path-integrated feature attribution.
- Each method page is intentionally a compact reference sheet: formula, symbol
  explanations, minimal code, visualization steps, and practical limits.

Known deviations from the source:

- The editable v0 uses a cleaner, more modular lecture design rather than
  pixel-copying the image-only slides.
- The real-data model uses Conv3D stages with a global-pooling classifier while
  preserving the stage2 CAM tap point used by the method pages.
- The main deck figures now use MSD Task06_Lung real CT tumour-present patches.
  Class 1 is a cropped CT patch containing tumour; class 0 is a non-tumour CT
  patch. Attribution maps are evaluated against the cropped tumour mask.
- The method code snippets are short teaching snippets. The executable source
  of truth remains `src/gradcam_repro/attribution.py`.

# Claim Spine

Thesis: attribution methods answer related but different questions. The deck
should teach them as a sequence of repairs: activation maps show what turns on,
class activation maps add class evidence, Grad-CAM removes CAM's architecture
constraint, later methods address coarse resolution, gradient fragility,
model-agnostic perturbation, and saturation.

Audience: technical readers who know CNNs and need a practical Grad-CAM method
reference.

Arc:

1. Define the attribution question and method taxonomy.
2. Start from activation maps: what is active before asking about a class.
3. Ground every method in one real MSD Lung CT patch experiment with tumour masks.
4. For every method, show the previous pain point, paper citation, formula,
   symbol definitions, minimal code, and visualization steps.
5. Compare methods by mechanism, cost, saturation behavior, and resolution.
6. Use generated figures and scores to make the deck reproducible.
7. Use the real CT experiment to keep the talk clinically grounded.

Method-slide contract:

- Every method page must include the paper/source line, or explicitly say that
  it is a control baseline rather than a paper method.
- Every method page must include a formula and symbol explanations.
- Every method page must include a small editable code snippet that maps back
  to `src/gradcam_repro/attribution.py`.
- Every method page must include visualization steps from input to overlay.
- Every method page must state which pain point it fixes from the previous
  method family.
- Demo images must come from the local experiment pipeline and the figure
  manifest, not from hand-drawn or copied screenshots.

Slide list:

1. Cover - why CNN attribution needs method comparison.
2. Roadmap - from activation maps to attribution, each method fixes a limitation.
3. Activation maps to CAM - active channels become class evidence.
4. Experimental setup - the real CT patch task and stage2 tap point.
5. notGradCAM - activation-only control with code, formula, symbols, steps.
6. Grad-CAM - class-discriminative reference with citation and steps.
7. Failure mode - channel-average signs can erase the map.
8. Guided Grad-CAM - voxel detail with sanity-check caveats.
9. LayerCAM - per-position gradient weighting for local evidence.
10. Occlusion - model-agnostic causal check, expensive by design.
11. Integrated Gradients - path-integral input attribution for saturation.
12. Integrated Grad-CAM - path-integral feature attribution.
13. Demo I - seven methods across three readable representative inputs.
14. Demo II - same input, true vs counterfactual class.
15. At a glance - method tradeoff table.
16. 3D notes - what changes when attribution is rendered over real volumes.
17. Practice - pick method combinations, not one winner.
18. References - chronological source list.
19. Discussion - decision questions for next iteration.

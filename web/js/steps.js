/* Per-method computation steps. Data, not logic.

   `term` names a key in that method's .decomp.json (or "path:N" for the Nth
   integration checkpoint, or "vector:alpha_all" for a per-channel bar strip).
   `sym` must match a data-sym that actually exists on that page's formula --
   these were read off the pages, not guessed, because a stale key makes the
   cross-highlight silently do nothing and that reads as a styling bug.

   Steps a method cannot supply are simply absent: the stepper drops any step
   whose term is missing from the payload, so no panel is ever a stand-in. */
export const STEPS = {
  // 4 formula anchors: A M mean up
  notgradcam: [
    { term: "activation", label: "Aᵏ", sym: "A",
      caption: "One channel of the tap, the busiest of the 640. This is all the baseline ever looks at." },
    { term: "vector:channel_means", label: "per-channel mean", sym: "M",
      caption: "Every channel's mean activation. The baseline weights all of them equally — no class enters here." },
    { term: "relu", label: "(1/K) Σₖ Aᵏ", sym: "mean",
      caption: "The plain mean over all 640 channels. It answers “where is the network active?”, not “where for this class?”" },
  ],

  // 8 formula anchors: A alpha gap grad L relu sum up
  gradcam: [
    { term: "activation", label: "Aᵏ", sym: "A",
      caption: "The tap activation for the channel with the largest |αₖ| — the one Grad-CAM leans on hardest." },
    { term: "vector:alpha_all", label: "αₖᶜ = GAP(∂yᶜ/∂Aᵏ)", sym: "gap",
      caption: "One pooled weight per channel. The gradient itself is spatially flat here, so this spread is where all the class information lives." },
    { term: "summed", label: "Σₖ αₖᶜAᵏ", sym: "sum",
      caption: "Weighted sum over all 640 channels. Still signed — blue is evidence against the class." },
    { term: "relu", label: "ReLU", sym: "relu",
      caption: "Negatives dropped, leaving evidence for the class only. This is the map the page scores, before it is upsampled 8× to the CT grid." },
  ],

  // 6 formula anchors: A grad hadamard L relu sum
  layercam: [
    { term: "activation", label: "Aᵏ", sym: "A",
      caption: "Same tap, same channel convention as Grad-CAM, so the two pages are comparable." },
    { term: "hadamard", label: "∂yᶜ/∂Aᵏ ⊙ Aᵏ", sym: "hadamard",
      caption: "LayerCAM weights per voxel instead of pooling first. On this network the gradient is spatially constant, so that freedom has nothing to add — the difference from Grad-CAM is only where the ReLU sits." },
    { term: "relu", label: "Σₖ ReLU(…)", sym: "relu",
      caption: "ReLU inside the channel sum rather than after it. That ordering is LayerCAM's actual contribution here." },
  ],

  // 4 formula anchors: H max pblank pcx
  occlusion: [
    { term: "scalar:intact", label: "P(c | x)", sym: "pcx",
      caption: "The intact probability, before anything is hidden." },
    { term: "occluded", label: "P(c | x blanked at p)", sym: "pblank",
      caption: "The score with a 16³ cube blanked at each position. No gradients anywhere in this method." },
    { term: "drop", label: "H(p)", sym: "H",
      caption: "The drop, intact minus occluded. A big drop means the network needed what was there." },
  ],

  // 7 formula anchors: alpha baseline grad ig integral riemann xi
  integrated_gradients: [
    { term: "path:0", label: "⅛ of the path", sym: "riemann",
      caption: "The Riemann sum barely started: a handful of steps from the baseline." },
    { term: "path:1", label: "¼", sym: "integral",
      caption: "Structure begins to separate from noise as more of the path accumulates." },
    { term: "path:2", label: "½", sym: "alpha",
      caption: "Halfway along the straight line from baseline to input." },
    { term: "path:3", label: "full integral", sym: "ig",
      caption: "The finished attribution. Comparing this with the earlier panels is what “saturation” actually looks like." },
  ],

  // 7 formula anchors: abs avg delta grad hadamard igc norm
  integrated_gradcam: [
    { term: "path:0", label: "⅛ of the path", sym: "avg",
      caption: "The integrated CAM after a few steps — the same path idea as Integrated Gradients, but accumulated at the tap." },
    { term: "path:1", label: "¼", sym: "delta",
      caption: "Each step contributes gradient × the activation difference from the baseline." },
    { term: "path:2", label: "½", sym: "hadamard",
      caption: "The map stabilises well before the path ends." },
    { term: "path:3", label: "full integral", sym: "igc",
      caption: "The finished integrated Grad-CAM at tap resolution." },
  ],

  // 6 formula anchors: abs gbp ggc hadamard norm upgc
  guided_gradcam: [
    { term: "guided_bp", label: "guided ∂yᶜ/∂x", sym: "gbp",
      caption: "Guided backpropagation at full input resolution: sharp, edge-level, and signed — but it does not know where the class is." },
    { term: "cam_upsampled", label: "ReLU(Lᶜ) upsampled", sym: "upgc",
      caption: "The Grad-CAM map, class-localised but coarse: 16³ at the tap, stretched 8× in plane to reach the CT grid." },
    { term: "product", label: "guided ⊙ Lᶜ", sym: "ggc",
      caption: "Their product. The CAM decides where, guided backprop decides which edges — neither alone is what this page scores." },
  ],
};

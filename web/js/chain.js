/* web/js/chain.js
   Single source of truth for the guided reading order, plus the shared
   top progress rail and bottom Prev/Next. Defined once; each page calls
   mountChainNav() with its own position. Zero deps, no build. */

export const CHAIN = [
  { file: "index.html",                        label: "Network",              group: "Network"   },
  { file: "methods/notgradcam.html",           label: "Baseline",             group: "Methods"   },
  { file: "methods/gradcam.html",              label: "Grad-CAM",             group: "Methods"   },
  { file: "methods/layercam.html",              label: "LayerCAM",             group: "Methods"   },
  { file: "methods/guided_gradcam.html",       label: "Guided Grad-CAM",      group: "Methods"   },
  { file: "methods/occlusion.html",            label: "Occlusion",            group: "Methods"   },
  { file: "methods/integrated_gradients.html", label: "Integrated Gradients", group: "Methods"   },
  { file: "methods/integrated_gradcam.html",   label: "Integrated Grad-CAM",  group: "Methods"   },
  { file: "compare.html",                      label: "Compare",              group: "Compare"   },
  { file: "reproduce.html",                    label: "Reproduce",            group: "Reproduce" },
];

export function mountChainNav(currentFile, base = "") {
  const i = CHAIN.findIndex((p) => p.file === currentFile);
  if (i < 0) throw new Error(`chain.js: unknown page "${currentFile}"`);
  const href = (p) => base + p.file;

  const rail = document.querySelector("[data-chain-rail]");
  if (rail) {
    rail.innerHTML = CHAIN.map((p, k) => {
      const cls = "chain-step" + (k === i ? " current" : k < i ? " done" : "");
      const cur = k === i ? ' aria-current="page"' : "";
      return `<a class="${cls}" href="${href(p)}"${cur}><span class="cs-dot"></span>` +
             `<span class="cs-lab">${p.label}</span></a>`;
    }).join("");
  }

  const foot = document.querySelector("[data-chain-foot]");
  if (foot) {
    const prev = CHAIN[i - 1], next = CHAIN[i + 1];
    foot.innerHTML =
      (prev ? `<a class="chain-prev" href="${href(prev)}">← ${prev.label}</a>`
            : `<span class="chain-prev-spacer"></span>`) +
      (next ? `<a class="chain-next" href="${href(next)}"><span class="cn-k">Next</span> ${next.label} →</a>`
            : `<span class="chain-end">End of the tour</span>`);
  }
}

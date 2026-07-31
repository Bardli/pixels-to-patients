/* Computation stepper: walk one method's real intermediate tensors, with the
   formula symbol highlighting in step.

   Three kinds of panel, because the arithmetic genuinely has three shapes:
     image   -- a 2D slice of a term (greyscale, or diverging when signed)
     vector  -- one value per channel (alpha across 640, or channel means)
     scalar  -- a single number (occlusion's intact probability)

   The vector panel exists for a measured reason. Under a globally pooled head
   the gradient at the tap is spatially constant, so a "gradient" image would be
   one flat colour and `weighted` would be pixel-identical to `activation`. The
   class-specific signal lives in the spread of the per-channel weights instead,
   so that is what gets drawn. See web/DATA_CONTRACT.md. */
import { loadJSON } from "./data.js?v=4";
import { divergeRGB, zeroAt, fmtNum } from "./diverge.js?v=1";
import { STEPS } from "./steps.js?v=1";

const ROOT = "../public/data";

function paintImage(canvas, payload) {
  const { shape, values, vmin, vmax, signed } = payload;
  const [h, w] = shape;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  const z = signed ? zeroAt(vmin, vmax) : 0;
  for (let i = 0; i < values.length; i++) {
    const t = values[i] / 255;
    let rgb;
    if (signed) {
      // Re-map so the payload's zero lands on the scale's midpoint, whatever
      // the asymmetry between vmin and vmax.
      const s = t < z ? (z > 0 ? (t / z) * 0.5 : 0.5)
                      : (z < 1 ? 0.5 + ((t - z) / (1 - z)) * 0.5 : 0.5);
      rgb = divergeRGB(s);
    } else {
      const v = Math.round(t * 255);
      rgb = [v, v, v];
    }
    img.data.set([rgb[0], rgb[1], rgb[2], 255], i * 4);
  }
  ctx.putImageData(img, 0, 0);
}

function paintVector(canvas, vec) {
  const vals = vec.values;
  const sorted = vals.slice().sort((a, b) => a - b);
  const n = sorted.length;
  const W = Math.min(640, n), H = 180;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, W, H);
  const lo = Math.min(0, vec.vmin), hi = Math.max(0, vec.vmax);
  const span = (hi - lo) || 1;
  const y0 = H - ((0 - lo) / span) * H;
  for (let i = 0; i < W; i++) {
    const v = sorted[Math.floor((i * n) / W)];
    const y = H - ((v - lo) / span) * H;
    const [r, g, b] = divergeRGB(v < 0 ? 0.15 : 0.85);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(i, Math.min(y, y0), 1, Math.max(1, Math.abs(y0 - y)));
  }
  ctx.strokeStyle = "rgba(154,160,166,0.55)";
  ctx.beginPath(); ctx.moveTo(0, y0); ctx.lineTo(W, y0); ctx.stroke();
}

/** Resolve a step's term name against the payload. Returns null if absent. */
function resolve(decomp, term) {
  if (term.startsWith("path:")) {
    const i = Number(term.slice(5));
    return decomp.path && decomp.path[i]
      ? { kind: "image", payload: decomp.path[i], frac: decomp.path[i].frac }
      : null;
  }
  if (term.startsWith("vector:")) {
    const v = decomp[term.slice(7)];
    return v && v.values ? { kind: "vector", payload: v } : null;
  }
  if (term.startsWith("scalar:")) {
    const v = decomp[term.slice(7)];
    return v == null ? null : { kind: "scalar", payload: v };
  }
  const t = decomp.terms && decomp.terms[term];
  return t ? { kind: "image", payload: t } : null;
}

export async function mountStepper(el, { method, exampleId, root = ROOT }) {
  const steps = STEPS[method];
  if (!steps) { el.remove(); return; }        // method defines no terms
  let decomp;
  try {
    decomp = await loadJSON(`${root}/examples/${exampleId}/attributions/${method}.decomp.json`);
  } catch {
    // No decomposition exported: remove the mount rather than draw a stand-in.
    // Saying nothing is better than illustrating something we did not compute.
    el.remove();
    return;
  }

  const available = steps
    .map((s) => ({ ...s, res: resolve(decomp, s.term) }))
    .filter((s) => s.res);
  if (!available.length) { el.remove(); return; }

  el.innerHTML =
    '<div class="stepper">' +
      '<div class="st-rail" role="tablist" aria-label="computation steps"></div>' +
      '<div class="st-stage">' +
        '<canvas class="st-canvas"></canvas>' +
        '<div class="st-scalar" hidden></div>' +
      "</div>" +
      '<p class="st-cap figcap"></p>' +
    "</div>";
  const rail = el.querySelector(".st-rail");
  const canvas = el.querySelector(".st-canvas");
  const scalar = el.querySelector(".st-scalar");
  const cap = el.querySelector(".st-cap");

  available.forEach((s, i) => {
    const b = document.createElement("button");
    b.className = "st-step"; b.type = "button";
    b.setAttribute("role", "tab");
    b.innerHTML = `<span class="st-n">${i + 1}</span><span class="st-l">${s.label}</span>`;
    b.addEventListener("click", () => show(i));
    rail.appendChild(b);
  });

  function show(i) {
    const s = available[i];
    const { kind, payload } = s.res;
    canvas.hidden = kind === "scalar";
    scalar.hidden = kind !== "scalar";
    canvas.classList.toggle("vector", kind === "vector");

    let detail = "";
    if (kind === "image") {
      paintImage(canvas, payload);
      detail = `${payload.shape[0]}×${payload.shape[1]}` +
        (payload.depth ? ` · z′ ${payload.feature_z}/${payload.depth}` : "") +
        ` · [${fmtNum(payload.vmin)}, ${fmtNum(payload.vmax)}]` +
        (payload.signed ? " · signed" : "");
    } else if (kind === "vector") {
      paintVector(canvas, payload);
      const neg = payload.values.filter((v) => v < 0).length;
      detail = `${payload.values.length} channels, sorted · ${neg} negative` +
        ` · [${fmtNum(payload.vmin)}, ${fmtNum(payload.vmax)}]`;
    } else {
      scalar.textContent = fmtNum(payload);
      detail = "a single number, not a map";
    }

    const ch = decomp.channel != null && kind === "image" && !s.term.startsWith("path:")
      ? ` · channel ${decomp.channel}` : "";
    cap.innerHTML = `<b>${i + 1}/${available.length}</b> ${s.caption}` +
      `<span class="st-meta">${detail}${ch}</span>`;

    rail.querySelectorAll(".st-step").forEach((b, j) =>
      b.setAttribute("aria-selected", String(j === i)));
    document.querySelectorAll("[data-sym]").forEach((n) =>
      n.classList.toggle("sym-on", n.dataset.sym === s.sym));
  }
  show(0);
}

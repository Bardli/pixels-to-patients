/* ============================================================
   Shared data + rendering helpers for the Grad-CAM/CT static site.
   No dependencies. ES module. Reads the web-export bundle
   (see web/DATA_CONTRACT.md).
   ============================================================ */

export async function loadJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch failed ${res.status}: ${url}`);
  return res.json();
}

/* Turbo colormap (same stops as the CSS --turbo accent). t in [0,1] -> [r,g,b]. */
const TURBO_STOPS = [
  [48, 18, 59], [65, 69, 171], [70, 117, 237], [57, 162, 252], [40, 194, 184],
  [91, 213, 78], [181, 208, 44], [249, 198, 49], [251, 128, 34], [234, 78, 27], [177, 25, 1],
];
export function turbo(t) {
  t = Math.max(0, Math.min(1, t));
  const seg = t * (TURBO_STOPS.length - 1);
  const i = Math.min(TURBO_STOPS.length - 2, Math.floor(seg));
  const f = seg - i;
  const a = TURBO_STOPS[i], b = TURBO_STOPS[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

/* Draw a grayscale slice payload {shape:[H,W], values:[0..255]} onto a canvas. */
export function drawGray(canvas, payload) {
  const [h, w] = payload.shape;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let k = 0; k < w * h; k++) {
    const v = payload.values[k];
    img.data[k * 4] = v; img.data[k * 4 + 1] = v; img.data[k * 4 + 2] = v; img.data[k * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/* Draw a heat overlay: turbo-coloured, alpha rising with intensity so cold areas stay clear. */
export function drawHeat(canvas, payload, { gamma = 1.6, maxAlpha = 210, floor = 0.10 } = {}) {
  const [h, w] = payload.shape;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let k = 0; k < w * h; k++) {
    const t = payload.values[k] / 255;
    const [r, g, b] = turbo(t);
    // Cold voxels stay transparent so the CT shows through; only hot regions paint.
    const a = t <= floor ? 0 : Math.pow((t - floor) / (1 - floor), gamma) * maxAlpha;
    img.data[k * 4] = r; img.data[k * 4 + 1] = g; img.data[k * 4 + 2] = b;
    img.data[k * 4 + 3] = Math.round(a);
  }
  ctx.putImageData(img, 0, 0);
}

/* Draw a feature-map channel: per-channel contrast-stretch (each channel's own
   max -> 1) then turbo-colour, so sparse post-ReLU activations still light up
   instead of rendering near-black. Structure is preserved; only contrast is
   stretched — noted in the caption. */
export function drawFeature(canvas, payload) {
  const [h, w] = payload.shape;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  let mx = 1;
  for (const v of payload.values) if (v > mx) mx = v;
  for (let k = 0; k < w * h; k++) {
    // gamma < 1 lifts sparse low-but-nonzero activations into the visible turbo range
    const [r, g, b] = turbo(Math.pow(payload.values[k] / mx, 0.6));
    img.data[k * 4] = r; img.data[k * 4 + 1] = g; img.data[k * 4 + 2] = b; img.data[k * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

export function realValue(payload, cell) {
  return payload.vmin + (payload.values[cell] / 255) * (payload.vmax - payload.vmin);
}

/* Attach pointer read-outs over a stack of aligned canvases.
   onCell(row, col, cell) fires with the hovered voxel index. */
export function attachHover(surface, shape, onCell, onLeave) {
  const [h, w] = shape;
  const handle = (ev) => {
    const rect = surface.getBoundingClientRect();
    const col = Math.min(w - 1, Math.max(0, Math.floor(((ev.clientX - rect.left) / rect.width) * w)));
    const row = Math.min(h - 1, Math.max(0, Math.floor(((ev.clientY - rect.top) / rect.height) * h)));
    onCell(row, col, row * w + col);
  };
  surface.addEventListener("pointermove", handle);
  surface.addEventListener("pointerleave", () => onLeave && onLeave());
}

/* Position a crosshair box (in %) over a WxH grid cell. */
export function cellBox(shape, row, col) {
  const [h, w] = shape;
  return { left: (col / w) * 100, top: (row / h) * 100, width: (1 / w) * 100, height: (1 / h) * 100 };
}

export const METHOD_LABELS = {
  notgradcam: "Mean activation map",
  gradcam: "Grad-CAM",
  guided_gradcam: "Guided Grad-CAM",
  layercam: "LayerCAM",
  occlusion: "Occlusion",
  integrated_gradients: "Integrated Gradients",
  integrated_gradcam: "Integrated Grad-CAM",
};

export const METRIC_LABELS = {
  mass_in_gt: "Mass in GT",
  inside_outside_ratio: "Inside / outside",
  pointing_acc: "Pointing hit",
};

export function fmtMetric(key, value) {
  if (key === "pointing_acc") return value >= 0.5 ? "✓ hit" : "✗ miss";
  if (key === "inside_outside_ratio") return value.toFixed(2) + "×";
  return (value * 100).toFixed(1) + "%";
}

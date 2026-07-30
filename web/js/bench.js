/* ============================================================
   Shared "reading bench" for every method page.
   initBench(method) wires: example tabs, CT + heat canvases,
   per-voxel hover read-out, verdict line, and the metric scorecard,
   all from the web-export bundle. See web/DATA_CONTRACT.md.
   ============================================================ */
import { loadJSON, drawGray, drawHeat, realValue, cellBox, fmtMetric, METRIC_LABELS, METHOD_LABELS } from "./data.js";

const METRIC_HINT = {
  enrichment: "vs a uniform map",
  mass_in_gt: "heat inside lesion",
  inside_outside_ratio: "contrast in vs out",
  pointing_acc: "peak on nodule",
};

export function initBench(method, root = "../public/data") {
  const $ = (id) => document.getElementById(id);
  const els = {
    tabs: $("tabs"), viewer: $("viewer"), ct: $("ct"), heat: $("heat"),
    cross: $("cross"), readout: $("readout"), verdict: $("verdict"), scorecard: $("scorecard"),
  };
  const label = METHOD_LABELS[method] || method;
  let cur = { ct: null, attr: null, shape: [1, 1] };

  const signOf = (id) => (/neg\d*$/.test(id) ? "−" : "+");
  const shortLabel = (id) => id.replace(/^\d+_/, "").replace(/_(pos|neg)\d*$/, "");

  async function select(id, tabEl) {
    for (const t of els.tabs.children) t.setAttribute("aria-selected", String(t === tabEl));
    const [meta, ct, attr] = await Promise.all([
      loadJSON(`${root}/examples/${id}/meta.json`),
      loadJSON(`${root}/examples/${id}/ct_slice.json`),
      loadJSON(`${root}/examples/${id}/attributions/${method}.json`),
    ]);
    cur = { ct, attr, shape: ct.shape };
    drawGray(els.ct, ct);
    drawHeat(els.heat, attr);
    if (els.gt) els.gt.src = `${root}/examples/${id}/mask_slice.png`;
    if (els.gtCap) els.gtCap.innerHTML =
      `<b class="gt-t">Ground truth</b> · the nodule`;

    // orientation: this is ONE axial slice of a D-deep 3D volume
    let badge = els.viewer.querySelector(".badge");
    if (!badge) { badge = document.createElement("div"); badge.className = "badge"; els.viewer.appendChild(badge); }
    badge.textContent = `axial · z ${meta.z_slice} / ${meta.input_shape[0]}`;

    const truth = meta.true_label === 1 ? "malignant" : "benign";
    const ok = meta.pred_label === meta.true_label;
    els.verdict.innerHTML =
      `case <b>${shortLabel(id)}</b> · truth <b>${truth}</b> · prediction ` +
      (ok ? `<span class="ok">correct</span>` : `<span class="no">wrong</span>`) +
      ` <span class="k">(logits ${meta.logits.map((v) => v.toFixed(2)).join(", ")})</span>`;

    const m = meta.metrics[method];
    els.scorecard.innerHTML = Object.keys(METRIC_LABELS).map((k) => {
      const cls = k === "pointing_acc" ? (m[k] >= 0.5 ? "gt" : "no") : "";
      return `<div class="metric"><div class="mk">${METRIC_LABELS[k]}</div>` +
             `<div class="mv ${cls}">${fmtMetric(k, m[k])}</div>` +
             `<div class="mh">${METRIC_HINT[k]}</div></div>`;
    }).join("");
  }

  els.viewer.addEventListener("pointermove", (ev) => {
    const [h, w] = cur.shape;
    const rect = els.viewer.getBoundingClientRect();
    const col = Math.min(w - 1, Math.max(0, Math.floor(((ev.clientX - rect.left) / rect.width) * w)));
    const row = Math.min(h - 1, Math.max(0, Math.floor(((ev.clientY - rect.top) / rect.height) * h)));
    const idx = row * w + col;
    const box = cellBox(cur.shape, row, col);
    Object.assign(els.cross.style, { left: box.left + "%", top: box.top + "%", width: box.width + "%", height: box.height + "%" });
    els.viewer.classList.add("hot");
    const ctv = cur.ct ? realValue(cur.ct, idx).toFixed(2) : "–";
    const av = cur.attr ? (cur.attr.values[idx] / 255).toFixed(2) : "–";
    els.readout.innerHTML =
      `<span><span class="k">voxel</span> <b>[${row}, ${col}]</b></span>` +
      `<span><span class="k">CT</span> <b>${ctv}</b></span>` +
      `<span><span class="k">${label}</span> <b>${av}</b></span>`;
  });
  els.viewer.addEventListener("pointerleave", () => {
    els.viewer.classList.remove("hot");
    els.readout.innerHTML = `<span class="k">hover the scan to read a voxel</span>`;
  });

  // Rebuild the bench into a side-by-side comparison: [ground truth | attribution].
  // The eye judges "did the heat land on the nodule?"; the scorecard quantifies it.
  function buildCompare() {
    const viewer = els.viewer;
    const bench = viewer.closest(".bench");
    if (bench) bench.classList.add("bench-compare");
    const parent = viewer.parentNode;
    const anchor = viewer.nextSibling;

    const gtFig = document.createElement("figure"); gtFig.className = "cmp";
    const gtBox = document.createElement("div"); gtBox.className = "viewer gt";
    const gtImg = document.createElement("img"); gtImg.id = "gtimg";
    gtImg.alt = "Ground-truth nodule region shaded green on the CT slice";
    gtBox.appendChild(gtImg); gtFig.appendChild(gtBox);
    const gtCap = document.createElement("figcaption");
    gtCap.innerHTML = `<b class="gt-t">Ground truth</b> · the nodule`;
    gtFig.appendChild(gtCap);

    const mFig = document.createElement("figure"); mFig.className = "cmp";
    mFig.appendChild(viewer);  // move the interactive viewer into the pair
    const mCap = document.createElement("figcaption");
    mCap.innerHTML = `<b>${label}</b> · where the model looked`;
    mFig.appendChild(mCap);

    const pair = document.createElement("div"); pair.className = "viewer-pair";
    pair.append(gtFig, mFig);
    parent.insertBefore(pair, anchor);
    els.gt = gtImg; els.gtCap = gtCap;

    const fc = parent.querySelector(".figcap");
    if (fc) fc.innerHTML =
      `<b>Fig.</b> same axial slice · <span class="gt">green</span> = ground-truth nodule · ` +
      `<span class="accent">${label}</span> heat-map on the right — hover it to read voxels`;
  }

  (async function init() {
    try {
      buildCompare();
      const manifest = await loadJSON(`${root}/manifest.json`);
      manifest.examples.forEach((id, i) => {
        const t = document.createElement("button");
        t.className = "tab"; t.setAttribute("role", "tab");
        t.innerHTML = `${shortLabel(id)} <span class="sign">${signOf(id)}</span>`;
        t.addEventListener("click", () => select(id, t));
        els.tabs.appendChild(t);
        if (i === 0) select(id, t);
      });
    } catch (err) {
      els.readout.innerHTML = `<span class="no">could not load data bundle — serve over http and run web-export</span>`;
      console.error(err);
    }
  })();
}

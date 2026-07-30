/* Channel-mean distribution histogram.
 *
 * Replaces the former 640-tile channel grid on methods/notgradcam.html. The
 * grid could not answer the question the section asks — "what is being
 * averaged?" — because 634 of its 640 tiles were text-only chips. One
 * distribution answers it.
 *
 * Log-spaced x bins: the means span 0.000 .. 0.342 with a median of 0.001, so a
 * linear axis collapses ~two thirds of the channels into the first bar and hides
 * the second mode entirely. Exact zeros cannot sit on a log axis, so they get
 * their own bar left of an axis break.
 *
 * Single series, single hue (--accent, validated >= 3:1 against --bg and above
 * the chroma floor), so no legend is needed — the title names it. The two modes
 * are direct-labelled; every bar carries a hover tooltip; a table view sits
 * behind a <details>.
 */

const NS = "http://www.w3.org/2000/svg";

const VB = { w: 760, h: 290 };
const PAD = { top: 40, right: 16, bottom: 44, left: 46 };
const GAP = 2; // surface gap between adjacent bars, per the mark spec
const RADIUS = 4; // rounded data-end
const BINS = 24;
const ZERO_SLOT = 40; // left gutter for the exact-zero annotation + axis break

function el(name, attrs = {}, text) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  if (text != null) node.textContent = text;
  return node;
}

/** Bar with only its top corners rounded, anchored to the baseline. */
function barPath(x, y, w, h, r) {
  const rr = Math.max(0, Math.min(r, w / 2, h));
  return `M${x} ${y + h}V${y + rr}a${rr} ${rr} 0 0 1 ${rr} ${-rr}h${w - 2 * rr}a${rr} ${rr} 0 0 1 ${rr} ${rr}V${y + h}Z`;
}

function niceTicks(max, count = 4) {
  if (max <= 0) return [0];
  const raw = max / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? 10 * mag;
  const ticks = [];
  for (let t = 0; t <= max + step * 0.001; t += step) ticks.push(t);
  return ticks;
}

const fmtMean = (v) => (v >= 0.01 ? v.toFixed(3) : v.toExponential(1).replace("e-", "e−"));

/**
 * @param {HTMLElement} mount
 * @param {{index:number, mean:number}[]} channels every channel at the tap
 * @param {{tapName?:string, caseId?:string}} meta
 */
export function renderChannelDistribution(mount, channels, meta = {}) {
  if (!Array.isArray(channels) || channels.length === 0) {
    throw new Error("renderChannelDistribution: no channels supplied");
  }
  mount.innerHTML = "";

  const means = channels.map((c) => c.mean);
  const zeros = means.filter((m) => m <= 0).length;
  const nonZero = means.filter((m) => m > 0).sort((a, b) => a - b);
  if (nonZero.length === 0) throw new Error("renderChannelDistribution: every channel is zero");

  const lo = Math.log10(nonZero[0]);
  const hi = Math.log10(nonZero[nonZero.length - 1]);
  const span = hi - lo || 1;
  const edges = Array.from({ length: BINS + 1 }, (_, i) => 10 ** (lo + (span * i) / BINS));
  const counts = new Array(BINS).fill(0);
  for (const m of nonZero) {
    const i = Math.min(BINS - 1, Math.floor(((Math.log10(m) - lo) / span) * BINS));
    counts[i] += 1;
  }

  const total = means.length;
  const sorted = [...means].sort((a, b) => a - b);
  const sum = sorted.reduce((a, b) => a + b, 0);
  const half = sorted.slice(0, Math.floor(total / 2)).reduce((a, b) => a + b, 0);
  const lowerHalfShare = sum > 0 ? (half / sum) * 100 : 0;

  const yMax = Math.max(zeros, ...counts);
  const plotW = VB.w - PAD.left - PAD.right - ZERO_SLOT;
  const plotH = VB.h - PAD.top - PAD.bottom;
  const x0 = PAD.left + ZERO_SLOT;
  const yBase = PAD.top + plotH;
  const barW = plotW / BINS;
  const yOf = (v) => yBase - (v / yMax) * plotH;

  const svg = el("svg", {
    viewBox: `0 0 ${VB.w} ${VB.h}`,
    // width/height as attributes too: the chart must size itself even if a
    // cached stylesheet has not picked up .chandist-svg yet.
    width: "100%",
    preserveAspectRatio: "xMidYMid meet",
    class: "chandist-svg",
    role: "img",
    "aria-label":
      `Distribution of mean activation across ${total} ${meta.tapName ?? "feature-map"} channels. ` +
      `Bimodal: one cluster near ${fmtMean(sorted[Math.floor(total / 2)])}, another near ` +
      `${fmtMean(sorted[Math.floor(total * 0.85)])}. ${zeros} channel${zeros === 1 ? "" : "s"} exactly zero.`,
  });

  // ---- recessive grid + y axis -------------------------------------------
  const gGrid = el("g", { class: "cd-grid" });
  for (const t of niceTicks(yMax)) {
    const y = yOf(t);
    gGrid.appendChild(el("line", { x1: PAD.left - 6, x2: VB.w - PAD.right, y1: y, y2: y }));
    gGrid.appendChild(el("text", { x: PAD.left - 10, y: y + 3.5, class: "cd-tick cd-tick-y" }, String(t)));
  }
  svg.appendChild(gGrid);
  svg.appendChild(el("text", { x: PAD.left - 10, y: PAD.top - 12, class: "cd-axis-title" }, "channels"));

  // ---- bars ---------------------------------------------------------------
  const gBars = el("g", { class: "cd-bars" });
  const bars = [];

  // Exact zeros get an annotation, not a bar. At 1 of 640 against a 282-tall
  // mode a bar would be a sub-pixel sliver — the wrong encoding for a count
  // this small, and it would imply a magnitude the reader cannot see.
  if (zeros > 0) {
    const zx = PAD.left + 4;
    svg.appendChild(el("line", { x1: zx, x2: zx, y1: yBase - 9, y2: yBase, class: "cd-base" }));
    svg.appendChild(
      el("text", { x: zx, y: yBase - 15, class: "cd-note", "text-anchor": "start" },
        `${zeros} exactly 0`),
    );
    svg.appendChild(el("text", { x: zx, y: VB.h - PAD.bottom + 16, class: "cd-tick", "text-anchor": "start" }, "0"));
    // axis break between the zero gutter and the log axis
    const bxg = PAD.left + ZERO_SLOT - 11;
    svg.appendChild(el("path", { d: `M${bxg} ${yBase + 5}l4 -7M${bxg + 5} ${yBase + 5}l4 -7`, class: "cd-break" }));
  }

  for (let i = 0; i < BINS; i += 1) {
    if (counts[i] === 0) continue;
    const w = barW - GAP;
    const h = yBase - yOf(counts[i]);
    bars.push({
      node: el("path", {
        d: barPath(x0 + i * barW + GAP / 2, yOf(counts[i]), w, h, RADIUS),
        class: "cd-bar",
      }),
      label: `${fmtMean(edges[i])} – ${fmtMean(edges[i + 1])}`,
      count: counts[i],
    });
  }
  for (const b of bars) gBars.appendChild(b.node);
  svg.appendChild(gBars);

  // ---- baseline + log x ticks --------------------------------------------
  svg.appendChild(el("line", { x1: PAD.left, x2: VB.w - PAD.right, y1: yBase, y2: yBase, class: "cd-base" }));
  for (let e = Math.ceil(lo); e <= Math.floor(hi); e += 1) {
    const x = x0 + ((e - lo) / span) * plotW;
    svg.appendChild(el("line", { x1: x, x2: x, y1: yBase, y2: yBase + 4, class: "cd-base" }));
    svg.appendChild(el("text", { x, y: VB.h - PAD.bottom + 16, class: "cd-tick" }, `10${sup(e)}`));
  }
  svg.appendChild(
    el("text", { x: x0 + plotW / 2, y: VB.h - 8, class: "cd-axis-title cd-axis-x" }, "mean activation (log)"),
  );

  // ---- direct labels on the two modes ------------------------------------
  // Labels sit in the top margin with a leader down to the bar, so they can
  // never collide with a tall mode. They name the cluster and carry no count:
  // a bin's count is not the cluster's size, and the tooltip and table view
  // already give exact numbers.
  const peak = (from, to) => {
    let best = -1, bi = from;
    for (let i = from; i < to; i += 1) if (counts[i] > best) { best = counts[i]; bi = i; }
    return bi;
  };
  const mid = Math.floor(BINS / 2);
  for (const [bi, text, anchor] of [
    [peak(0, mid), "near-silent", "start"],
    [peak(mid, BINS), "active", "end"],
  ]) {
    if (counts[bi] === 0) continue;
    const cx = x0 + bi * barW + barW / 2;
    const ly = PAD.top - 20;
    svg.appendChild(el("line", { x1: cx, x2: cx, y1: ly + 5, y2: yOf(counts[bi]) - 3, class: "cd-leader" }));
    svg.appendChild(
      el("text", { x: cx + (anchor === "end" ? -5 : 5), y: ly, class: "cd-note", "text-anchor": anchor }, text),
    );
  }

  mount.appendChild(svg);

  // ---- hover tooltip ------------------------------------------------------
  const tip = document.createElement("div");
  tip.className = "cd-tip";
  tip.hidden = true;
  mount.appendChild(tip);
  for (const b of bars) {
    b.node.addEventListener("pointerenter", () => {
      tip.hidden = false;
      tip.innerHTML = `<b>${b.count}</b> channel${b.count === 1 ? "" : "s"}<br><span>${b.label}</span>`;
    });
    b.node.addEventListener("pointermove", (ev) => {
      const r = mount.getBoundingClientRect();
      tip.style.left = `${ev.clientX - r.left}px`;
      tip.style.top = `${ev.clientY - r.top}px`;
    });
    b.node.addEventListener("pointerleave", () => { tip.hidden = true; });
  }

  // ---- table view ---------------------------------------------------------
  const det = document.createElement("details");
  det.className = "cd-table";
  det.appendChild(Object.assign(document.createElement("summary"), { textContent: "Table view" }));
  const rows = bars.map((b) => `<tr><td>${b.label}</td><td>${b.count}</td></tr>`).join("");
  det.insertAdjacentHTML(
    "beforeend",
    `<table><thead><tr><th>mean activation</th><th>channels</th></tr></thead><tbody>${rows}</tbody></table>`,
  );
  mount.appendChild(det);

  return { total, zeros, lowerHalfShare, median: sorted[Math.floor(total / 2)] };
}

function sup(n) {
  const map = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  return String(n).split("").map((c) => map[c] ?? c).join("");
}

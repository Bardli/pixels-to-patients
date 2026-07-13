/* ============================================================
   3D forward-pass scene + fan-out feature deck.
   CSS-3D (transform-style: preserve-3d) — no library, no build.
   Consumes the web-export bundle (see web/DATA_CONTRACT.md).
   Progressive enhancement: without JS the first scene shows as a
   static figure; under prefers-reduced-motion the scroll
   choreography is disabled and a single representative beat is shown.
   ============================================================ */
import { drawGray, drawHeat, drawFeature } from "./data.js?v=2";

const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const REDUCE = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* Build a .plane holding one or more canvases/images, indexed for translateZ. */
function makePlane(i, children) {
  const p = document.createElement("div");
  p.className = "plane";
  p.style.setProperty("--i", i);
  for (const c of children) p.appendChild(c);
  return p;
}

function canvasFor(payload, kind) {
  const cv = document.createElement("canvas");
  if (kind === "heat") drawHeat(cv, payload);
  else if (kind === "feature") drawFeature(cv, payload);
  else drawGray(cv, payload);
  return cv;
}

/* Drag-to-orbit. Writes rx/ry through `apply(dx, dy)`; caller owns the state. */
function attachOrbit(stage, apply) {
  let dragging = false, px = 0, py = 0;
  stage.addEventListener("pointerdown", (e) => {
    dragging = true; px = e.clientX; py = e.clientY;
    stage.classList.add("grabbing");
    try { stage.setPointerCapture(e.pointerId); } catch (_) {}
  });
  stage.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    apply(e.clientX - px, e.clientY - py);
    px = e.clientX; py = e.clientY;
  });
  const end = () => { dragging = false; stage.classList.remove("grabbing"); };
  stage.addEventListener("pointerup", end);
  stage.addEventListener("pointercancel", end);
  stage.addEventListener("pointerleave", end);
}

/* ------------------------------------------------------------------
   Component C — fan-out feature deck. Renders one activations entry
   (its top channels) as a stack of planes you can orbit and fan apart.
   ------------------------------------------------------------------ */
export function mountDeck(mount, entry, { badge } = {}) {
  mount.innerHTML = "";
  const n = entry.channels.length;

  const stage = document.createElement("div");
  stage.className = "stage3d deck-solo";
  const deck = document.createElement("div");
  deck.className = "deck3d";
  deck.style.setProperty("--n", n);
  deck.style.setProperty("--mid", (n - 1) / 2);
  deck.style.setProperty("--gap", "48px");

  const scene = document.createElement("div");
  scene.className = "beat-scene on";
  scene.style.setProperty("--mid", (n - 1) / 2);
  entry.channels.forEach((c, i) => {
    scene.appendChild(makePlane(i, [canvasFor(c.slice, "feature")]));
  });
  deck.appendChild(scene);
  stage.appendChild(deck);

  if (badge) {
    const b = document.createElement("div");
    b.className = "badge";
    b.textContent = badge;
    stage.appendChild(b);
  }
  mount.appendChild(stage);

  const state = { rx: 17, ry: -27 };
  const apply = () => {
    deck.style.setProperty("--rx", clamp(state.rx, -6, 42) + "deg");
    deck.style.setProperty("--ry", state.ry + "deg");
  };
  apply();
  if (!REDUCE) {
    attachOrbit(stage, (dx, dy) => { state.ry += dx * 0.4; state.rx -= dy * 0.3; apply(); });
  }

  // Deck is fixed fully fanned (--gap 48px, set above); the flat↔fanned slider was removed by design.
}

/* ------------------------------------------------------------------
   Component B — scroll-driven 3D pipeline. One CT volume walks the
   network beat by beat; drag to orbit, scroll to advance.
   ------------------------------------------------------------------ */
function planeScene({ tap = false } = {}) {
  const s = document.createElement("div");
  s.className = "beat-scene" + (tap ? " tap" : "");
  return s;
}

function fillChannels(scene, channels) {
  const n = channels.length;
  scene.style.setProperty("--mid", (n - 1) / 2);
  scene.style.setProperty("--gap", "20px");
  channels.forEach((c, i) => scene.appendChild(makePlane(i, [canvasFor(c.slice, "feature")])));
}

function buildScenes(deck, { acts, ctPayload, heatPayload }) {
  const find = (l) => acts.find((e) => e.layer === l);
  const shape = (e) => e ? `${e.feature_shape[0]}×${e.feature_shape.slice(1).join("×")}` : "—";
  const shown = (e) => e ? `${e.channels.length} of ${e.feature_shape[0]}` : "—";
  const s1 = find("stage1"), s2 = find("stage2"), s3 = find("stage3");
  const scenes = [];

  // 0 — input volume (single channel)
  {
    const el = planeScene();
    el.style.setProperty("--mid", 0);
    el.style.setProperty("--gap", "0px");
    el.appendChild(makePlane(0, [canvasFor(ctPayload, "gray")]));
    scenes.push({ el, rail: "Input", title: "CT input volume",
      body: "One axial CT patch enters as a single-channel volume (1×D×H×W) — the only thing the network ever sees." });
  }
  // 1 — stage1
  { const el = planeScene(); fillChannels(el, s1.channels);
    scenes.push({ el, rail: "stage1", title: "stage1 · early features",
      body: `Conv block 1 lifts the scan into ${shape(s1)} — low-level edges and texture. Showing ${shown(s1)} channels.` }); }
  // 2 — stage2 (CAM tap)
  { const el = planeScene({ tap: true }); fillChannels(el, s2.channels);
    scenes.push({ el, rail: "stage2 · tap", title: "stage2 — the CAM tap", tap: true,
      body: `Every attribution method reads the feature volume here (${shape(s2)}). Grad-CAM weights these channels by their gradient toward the class score.` }); }
  // 3 — stage3
  { const el = planeScene(); fillChannels(el, s3.channels);
    scenes.push({ el, rail: "stage3", title: "stage3 · deep features",
      body: `Conv block 3 — the deepest, most class-specific features (${shape(s3)}), just before pooling.` }); }
  // 4 — global pool -> logit
  {
    const el = document.createElement("div");
    el.className = "beat-scene";
    const bars = document.createElement("div");
    bars.className = "bars3d";
    const means = s3.channels.map((c) => c.mean);
    const lo = Math.min(...means), hi = Math.max(...means, lo + 1e-6);
    means.forEach((m) => {
      const b = document.createElement("div");
      b.className = "bar";
      b.style.height = (18 + ((m - lo) / (hi - lo)) * 82) + "%";
      bars.appendChild(b);
    });
    const chip = document.createElement("div");
    chip.className = "logit-chip";
    chip.innerHTML = 'GAP → logit <span class="g">→ p(tumour)</span>';
    el.appendChild(bars); el.appendChild(chip);
    scenes.push({ el, rail: "GAP → logit", title: "Global pool → class score",
      body: "Each feature channel is averaged to a single number; a linear layer turns that vector into the tumour-vs-not logit." });
  }
  // 5 — Grad-CAM back-projection
  {
    const el = planeScene();
    el.style.setProperty("--mid", 0);
    el.style.setProperty("--gap", "0px");
    el.appendChild(makePlane(0, [canvasFor(ctPayload, "gray"), canvasFor(heatPayload, "heat")]));
    scenes.push({ el, rail: "Grad-CAM", title: "Project the evidence back",
      body: "Grad-CAM upsamples the gradient-weighted stage2 map back onto the scan — a picture of where the class score came from." });
  }

  scenes.forEach((s) => deck.appendChild(s.el));
  return scenes;
}

export function initPipeline({ stage, deck, rail, cap, tapFlag, prevBtn, nextBtn, playBtn, acts, ctPayload, heatPayload }) {
  const scenes = buildScenes(deck, { acts, ctPayload, heatPayload });

  rail.innerHTML = scenes.map((s, i) =>
    `<button class="step" data-i="${i}" aria-label="Go to ${s.rail}"><span class="dot"></span><span class="st-lab">${s.rail}</span></button>`).join("");
  const steps = [...rail.children];

  let active = -1;
  const setBeat = (i) => {
    i = clamp(i, 0, scenes.length - 1);
    if (i === active) return;
    active = i;
    scenes.forEach((s, k) => {
      s.el.classList.toggle("on", k === i);
      s.el.classList.toggle("prev", k < i);
    });
    steps.forEach((st, k) => {
      st.classList.toggle("active", k === i);
      st.classList.toggle("done", k < i);
    });
    cap.innerHTML = `<b>${scenes[i].title}.</b> ${scenes[i].body}`;
    tapFlag.hidden = !scenes[i].tap;
  };

  // fixed, pleasant viewing angle; drag adds an offset on top (no scroll coupling)
  const state = { ry: -20, rx: 14, dry: 0, drx: 0 };
  const applyRot = () => {
    deck.style.setProperty("--ry", (state.ry + state.dry) + "deg");
    deck.style.setProperty("--rx", clamp(state.rx + state.drx, -6, 40) + "deg");
  };
  applyRot();
  setBeat(0);
  if (!REDUCE) attachOrbit(stage, (dx, dy) => { state.dry += dx * 0.4; state.drx -= dy * 0.3; applyRot(); });

  // advance beats by click / step / autoplay — no scroll hijacking
  let timer = 0;
  const stopPlay = () => { if (timer) { clearInterval(timer); timer = 0; if (playBtn) playBtn.textContent = "▶ Play"; } };
  const startPlay = () => {
    if (playBtn) playBtn.textContent = "❚❚ Pause";
    timer = setInterval(() => setBeat(active >= scenes.length - 1 ? 0 : active + 1), 1700);
  };
  const jump = (i) => { stopPlay(); setBeat(i); };
  steps.forEach((st, k) => st.addEventListener("click", () => jump(k)));
  if (prevBtn) prevBtn.addEventListener("click", () => jump(active - 1));
  if (nextBtn) nextBtn.addEventListener("click", () => jump(active + 1));
  if (playBtn) playBtn.addEventListener("click", () => (timer ? stopPlay() : startPlay()));
}

/* ------------------------------------------------------------------
   3D architecture — an isometric CNN "block" diagram. Each layer is a
   stack of thin slabs (a "loaf"): face size encodes spatial resolution,
   slab count encodes channel depth, so you watch resolution shrink as
   channels grow. Blocks slide/assemble in when the section scrolls into
   view (one-shot, staggered); drag to orbit. Conv stages are clickable
   to drive the activation inspector below. No scroll-scrub track.
   ------------------------------------------------------------------ */
export function mountArchitecture(mount, graph, { onSelect } = {}) {
  mount.innerHTML = "";
  const nodes = graph.nodes;
  const N = nodes.length;
  const CONV = new Set(["stage1", "stage2", "stage3"]);
  const maxC = Math.max(...nodes.map((n) => n.out_shape[0] || 1), 1);

  const stage = document.createElement("div");
  stage.className = "stage3d arch3d";
  stage.setAttribute("role", "img");
  stage.setAttribute("aria-label", "3D block diagram of the CNN; each block is a layer sized by its tensor shape. Drag to rotate.");
  const deck = document.createElement("div");
  deck.className = "arch-deck";
  stage.appendChild(deck);
  mount.appendChild(stage);

  const sw = stage.clientWidth || 900;
  const SLOT = Math.min(160, Math.max(46, (sw - 150) / Math.max(1, N - 1)));

  const clickable = typeof onSelect === "function";   // homepage passes none → display-only
  const blocks = [];
  nodes.forEach((n, i) => {
    const shape = n.out_shape;
    const C = shape[0] || 1;
    const spatial = shape.length >= 3 ? shape[shape.length - 1] : 1;
    const fs = Math.round(11 + Math.min(spatial, 32) * 3.3);                 // face px ← spatial
    const nSlabs = Math.max(1, Math.min(16, Math.round((C / maxC) * 16)));   // depth ← channels
    const isConv = CONV.has(n.id);

    const block = document.createElement("div");
    block.className = "block" + (isConv && clickable ? " clk" : "") + (n.cam_tap ? " tap" : "");
    block.style.setProperty("--x", ((i - (N - 1) / 2) * SLOT) + "px");
    block.style.setProperty("--fs", fs + "px");
    block.style.setProperty("--mid", (nSlabs - 1) / 2);
    block.style.setProperty("--sg", (fs > 46 ? 3.5 : 5) + "px");
    block.style.setProperty("--d", (i * 0.09) + "s");
    for (let k = 0; k < nSlabs; k++) {
      const s = document.createElement("div");
      s.className = "slab";
      s.style.setProperty("--i", k);
      block.appendChild(s);
    }
    if (isConv && clickable) block.addEventListener("click", () => onSelect(n.id));
    deck.appendChild(block);
    blocks.push({ el: block, id: n.id, isConv });
  });

  const labs = document.createElement("div");
  labs.className = "arch-labels";
  labs.innerHTML = nodes.map((n) => {
    const layer = CONV.has(n.id) ? n.id : "";
    const pr = n.param_count ? n.param_count.toLocaleString("en-US") + " params" : "—";
    return `<div class="arch-lab${n.cam_tap ? " tap" : ""}" data-layer="${layer}">` +
      `<b>${n.name}</b><span class="sh">${n.out_shape.join("×")}</span><span class="pr">${pr}</span></div>`;
  }).join("");
  mount.appendChild(labs);

  const st = { rx: 15, ry: -18 };
  const apply = () => {
    deck.style.setProperty("--rx", clamp(st.rx, -4, 40) + "deg");
    deck.style.setProperty("--ry", st.ry + "deg");
  };
  apply();
  if (!REDUCE) attachOrbit(stage, (dx, dy) => { st.ry += dx * 0.4; st.rx -= dy * 0.3; apply(); });

  let revealed = false;
  const reveal = () => { if (revealed) return; revealed = true; blocks.forEach((b) => b.el.classList.add("in")); };
  const inView = () => { const r = stage.getBoundingClientRect(); return r.top < window.innerHeight * 0.92 && r.bottom > 0; };
  if (REDUCE || !("IntersectionObserver" in window)) {
    reveal();
  } else {
    const io = new IntersectionObserver((es) => {
      for (const e of es) if (e.isIntersecting) { reveal(); io.disconnect(); break; }
    }, { threshold: 0.12 });
    io.observe(stage);
    if (inView()) reveal();                       // already visible at mount → animate now
    const onScroll = () => { if (inView()) { reveal(); window.removeEventListener("scroll", onScroll); } };
    window.addEventListener("scroll", onScroll, { passive: true });   // never stay invisible
  }

  return {
    select(layerId) {
      blocks.forEach((b) => b.el.classList.toggle("sel", b.isConv && b.id === layerId));
      labs.querySelectorAll(".arch-lab").forEach((l) => l.classList.toggle("sel", l.dataset.layer === layerId));
    },
  };
}

/* ------------------------------------------------------------------
   Homepage flow — the running example's REAL activation maps at each
   layer, connected left→right by lines. Hover a layer to enlarge it and
   unfold its top channels. A concrete, example-driven alternative to the
   abstract 3D block diagram (which lives on the Network page).
   ------------------------------------------------------------------ */
export function mountFlow(mount, { acts, ctPayload, heatPayload }) {
  mount.innerHTML = "";
  const find = (l) => acts.find((e) => e.layer === l);
  const shapeOf = (e) => (e ? e.feature_shape.join("×") : "—");
  const specs = [
    { name: "CT input", sub: "1×32×32×32", tiles: [{ p: ctPayload, k: "gray" }], hint: "the scan going in" },
    { name: "stage1", sub: shapeOf(find("stage1")), feat: find("stage1"), hint: "edges & texture" },
    { name: "stage2", sub: shapeOf(find("stage2")), feat: find("stage2"), tap: true, hint: "the CAM tap — every method reads here" },
    { name: "stage3", sub: shapeOf(find("stage3")), feat: find("stage3"), hint: "deep, class-specific features" },
    { name: "Grad-CAM", sub: "where it looked", tiles: [{ p: ctPayload, k: "gray" }, { p: heatPayload, k: "heat" }], hint: "the class evidence, projected back" },
  ];

  const flow = document.createElement("div");
  flow.className = "flow";
  specs.forEach((spec, i) => {
    if (i > 0) {
      const link = document.createElement("div");
      link.className = "flow-link";
      flow.appendChild(link);
    }
    flow.appendChild(buildFlowItem(spec));
  });
  mount.appendChild(flow);
}

function buildFlowItem(spec) {
  const item = document.createElement("div");
  item.className = "flow-item" + (spec.tap ? " tap" : "");

  const tile = document.createElement("div");
  tile.className = "flow-tile";
  if (spec.feat) tile.appendChild(canvasFor(spec.feat.channels[0].slice, "feature"));
  else for (const t of spec.tiles) tile.appendChild(canvasFor(t.p, t.k));
  item.appendChild(tile);

  const name = document.createElement("div");
  name.className = "flow-name";
  name.innerHTML = `<b>${spec.name}</b><span>${spec.sub}</span>`;
  item.appendChild(name);

  const pop = document.createElement("div");
  pop.className = "flow-pop";
  const grid = document.createElement("div");
  grid.className = "pop-grid";
  if (spec.feat) {
    spec.feat.channels.forEach((c) => {
      const cell = document.createElement("div");
      cell.className = "pop-cell";
      cell.appendChild(canvasFor(c.slice, "feature"));
      const lab = document.createElement("span");
      lab.textContent = "ch " + c.index;
      cell.appendChild(lab);
      grid.appendChild(cell);
    });
  } else {
    const cell = document.createElement("div");
    cell.className = "pop-cell wide";
    for (const t of spec.tiles) cell.appendChild(canvasFor(t.p, t.k));
    grid.appendChild(cell);
  }
  pop.appendChild(grid);
  const cap = document.createElement("div");
  cap.className = "pop-cap";
  cap.innerHTML = spec.feat
    ? `<b>${spec.name}</b> · top ${spec.feat.channels.length} of ${spec.feat.feature_shape[0]} channels`
    : `<b>${spec.name}</b> · ${spec.hint}`;
  pop.appendChild(cap);
  item.appendChild(pop);
  return item;
}

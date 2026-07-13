/* web/js/mathlegend.js
   Links a .mathblock's symbols to its .math-where legend. Hovering or
   focusing any [data-sym] element highlights every element sharing that
   data-sym across BOTH the formula and the legend (bidirectional).
   Pure progressive enhancement: without JS, formula + legend stay fully
   readable static content. No deps, no build. */
export function wireMathLegend(root = document) {
  root.querySelectorAll(".mathblock").forEach((block) => {
    // the legend is the nearest following .math-where sibling
    let legend = block.nextElementSibling;
    while (legend && !legend.classList.contains("math-where")) legend = legend.nextElementSibling;
    if (!legend) return;

    const scopes = [block, legend];
    const set = (sym, on) => {
      if (!sym) return;
      for (const s of scopes)
        s.querySelectorAll(`[data-sym="${sym}"]`).forEach((el) => el.classList.toggle("sym-hot", on));
    };
    const bind = (el) => {
      const sym = el.dataset.sym;
      el.tabIndex = 0;
      el.addEventListener("pointerenter", () => set(sym, true));
      el.addEventListener("pointerleave", () => set(sym, false));
      el.addEventListener("focus", () => set(sym, true));
      el.addEventListener("blur", () => set(sym, false));
    };
    for (const s of scopes) s.querySelectorAll("[data-sym]").forEach(bind);
  });
}

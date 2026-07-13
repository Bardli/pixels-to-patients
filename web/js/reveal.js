/* Scroll-reveal: fade .reveal elements in. Elements already in view on load
   are revealed immediately (no dependency on observer timing); the rest fade
   in as they enter the viewport. Degrades to "always visible". */
(function () {
  const els = [...document.querySelectorAll(".reveal")];
  if (!els.length) return;
  const reveal = (el) => el.classList.add("in");
  if (!("IntersectionObserver" in window)) { els.forEach(reveal); return; }
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting) { reveal(e.target); io.unobserve(e.target); }
    }
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });
  for (const el of els) {
    const r = el.getBoundingClientRect();
    if (r.top < window.innerHeight && r.bottom > 0) reveal(el); // already visible on load
    else io.observe(el);
  }
})();

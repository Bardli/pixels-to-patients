/* Signed colour scale. Gradients and channel-summed maps are the only signed
   quantities the site shows, and painting them with turbo would make a large
   negative look like a large positive -- which is exactly the failure these
   pages are trying to teach, since negative weights are what let the closing
   ReLU erase part of a map. Blue = negative, white = zero, red = positive.
   Kept out of data.js so neither scale grows a mode flag. */
export function divergeRGB(t) {
  const u = Math.max(0, Math.min(1, t));
  const s = (u - 0.5) * 2;                       // -1 .. 1
  const m = Math.abs(s);
  const lo = [42, 82, 152], hi = [176, 58, 46], mid = [246, 244, 238];
  const end = s < 0 ? lo : hi;
  return [0, 1, 2].map((i) => Math.round(mid[i] + (end[i] - mid[i]) * m));
}

/** Where zero sits in a payload's 0..1 range, for centring the scale. */
export function zeroAt(vmin, vmax) {
  if (vmax <= vmin) return 0.5;
  return Math.max(0, Math.min(1, (0 - vmin) / (vmax - vmin)));
}

/** Format a number for a caption without pretending to precision it lacks. */
export function fmtNum(v) {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a < 1e-3 || a >= 1e5) return v.toExponential(2);
  return v.toFixed(a < 1 ? 4 : 3);
}

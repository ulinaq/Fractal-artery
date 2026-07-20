// perfusion — viridis colour ramp (perceptual, colour-blind safe).
// Shared by the 2-D workspace (render.js) and the 3-D viewer (viewer3d.js).
export const VIRIDIS = [
  [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
  [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
];

// interpolated [r, g, b] (0-255 floats) for t in [0, 1]
export function viridis(t) {
  t = Math.max(0, Math.min(1, t)) * (VIRIDIS.length - 1);
  const i = Math.floor(t), f = t - i;
  const a = VIRIDIS[i], b = VIRIDIS[Math.min(i + 1, VIRIDIS.length - 1)];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

// same, as a CSS `rgb(...)` string
export function viridisCss(t) {
  const [r, g, b] = viridis(t);
  return `rgb(${r | 0},${g | 0},${b | 0})`;
}

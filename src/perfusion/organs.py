"""Organ-shaped tissue domains (service layer): 2-D outlines and 3-D masks.

Stylised, *connected* organ geometries for growing trees into and solving on —
used by the Tree Explorer and as tissue presets in the 2-D workspace. Each shape
is defined in unit coordinates (roughly ``[-1, 1]``) and mapped into a physical
window: organs keep their proportions (uniform scale), while the generic
``rectangle``/``ellipse``/``box``/``ellipsoid`` stretch to fill it.

Every shape is a single connected region (a split tissue mask would leave a
singular Darcy block — see ``runner.validate_run``); ``tests/test_organs.py``
checks that on the default grids. Pure NumPy, no web or optional deps.
"""

from __future__ import annotations

import numpy as np

# name -> human label; the order is the dropdown order
SHAPES_2D = {
    "ellipse": "ellipse", "rectangle": "rectangle", "wedge": "wedge (fan)",
    "heart": "heart (ventricles)", "kidney": "kidney", "brain": "brain (lateral)", "liver": "liver",
    "lung": "lung", "tongue": "tongue", "tumour": "tumour (irregular)",
    "placenta": "placenta (disc)",
}
SHAPES_3D = {
    "ellipsoid": "ellipsoid", "box": "box", "cylinder": "cylinder",
    "hemisphere": "hemisphere (dome)", "heart": "heart (myocardial wall)", "kidney": "kidney",
    "brain": "brain (two hemispheres)", "liver": "liver", "lung": "lung",
    "tumour": "tumour (irregular)",
}
STRETCH = {"ellipse", "rectangle", "wedge", "box", "ellipsoid", "cylinder", "hemisphere"}


def _chaikin(pts, rounds=2):
    """Corner-cutting smoothing of a closed polygon."""
    p = np.asarray(pts, float)
    for _ in range(rounds):
        q = np.roll(p, -1, axis=0)
        p = np.column_stack([0.75 * p + 0.25 * q, 0.25 * p + 0.75 * q]).reshape(-1, 2)
    return p


def _unit_outline(name: str, n: int = 32, seed: int = 0) -> np.ndarray:
    """A closed outline in unit coordinates (counter-clockwise)."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if name == "ellipse":
        return np.column_stack([np.cos(t), np.sin(t)])
    if name == "rectangle":
        return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float)
    if name == "wedge":
        return np.array([[-0.25, -1], [0.25, -1], [1, 1], [-1, 1]], float)
    if name == "kidney":                     # an ellipse with a hilum notch
        r = 1 - 0.32 * np.exp(-(np.angle(np.exp(1j * (t - np.pi))) ** 2) / 0.18)
        return np.column_stack([0.62 * r * np.cos(t), r * np.sin(t)])
    if name == "brain":                      # cerebrum with gyri + a cerebellum bump
        r = 1 + 0.035 * np.sin(13 * t)
        x, y = 1.0 * r * np.cos(t), 0.72 * r * np.sin(t)
        lump = np.exp(-((t - 5.0) ** 2) / 0.08)          # lower-back (cerebellum)
        return np.column_stack([x + 0.10 * lump, y - 0.18 * lump])
    if name == "tumour":                     # a seeded, lobulated blob
        rng = np.random.default_rng(seed)
        k = np.arange(2, 7)
        amp, ph = rng.uniform(0.04, 0.12, k.size), rng.uniform(0, 2 * np.pi, k.size)
        r = 0.85 + (amp[:, None] * np.sin(k[:, None] * t + ph[:, None])).sum(0)
        return np.column_stack([r * np.cos(t), r * np.sin(t)])
    if name == "placenta":                   # a flattened disc
        return np.column_stack([np.cos(t), 0.42 * np.sin(t) * (1 + 0.15 * np.cos(t))])
    hand = {                                 # hand-drawn silhouettes, then smoothed
        # frontal cardiac silhouette: right atrium (left of image), flat inferior
        # (diaphragmatic) border, apex at lower right, left-ventricular border,
        # left-atrial appendage / pulmonary trunk and the great-vessel base on top
        "heart": [(-.52, .58), (-.74, .34), (-.8, .02), (-.72, -.3), (-.5, -.5),
                  (-.1, -.6), (.32, -.68), (.66, -.78), (.84, -.56), (.9, -.2),
                  (.8, .14), (.6, .36), (.44, .56), (.16, .66), (-.22, .66)],
        "liver": [(-1, .30), (-.75, .72), (-.15, .88), (.45, .74), (1, .46), (.9, .22),
                  (.42, .06), (.02, -.16), (-.42, -.58), (-.8, -.66), (-1, -.22)],
        "lung": [(.05, 1), (.35, .72), (.55, .25), (.62, -.35), (.5, -.85), (.05, -.95),
                 (-.35, -.8), (-.32, -.45), (-.12, -.25), (-.2, .05), (-.12, .55)],
        "tongue": [(-.45, -1), (.45, -1), (.55, -.2), (.5, .45), (.32, .85), (0, 1),
                   (-.32, .85), (-.5, .45), (-.55, -.2)],
    }
    if name in hand:
        return _chaikin(hand[name], rounds=2)
    raise ValueError(f"unknown 2-D shape {name!r}; choose from {list(SHAPES_2D)}")


def polygon(name: str, domain, size: float = 0.85, seed: int = 0) -> list:
    """The 2-D outline of ``name`` placed in ``domain = (x0, x1, y0, y1)``."""
    x0, x1, y0, y1 = domain
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hx, hy = size * (x1 - x0) / 2, size * (y1 - y0) / 2
    u = _unit_outline(name, seed=seed)
    if name not in STRETCH:                  # organs keep their proportions
        hx = hy = min(hx, hy)
    return [[float(cx + hx * a), float(cy + hy * b)] for a, b in u]


def _unit_inside_3d(name, u, v, w, seed=0):
    """Boolean inside-test for a 3-D shape in unit coordinates (w is 'up')."""
    if name == "box":
        return (np.abs(u) <= 1) & (np.abs(v) <= 1) & (np.abs(w) <= 1)
    if name == "ellipsoid":
        return u**2 + v**2 + w**2 <= 1
    if name == "cylinder":
        return (u**2 + v**2 <= 1) & (np.abs(w) <= 1)
    if name == "hemisphere":                 # dome on the floor of the box
        return u**2 + v**2 + ((w + 1) / 2) ** 2 <= 1
    if name == "heart":                      # anatomical heart: the myocardial wall
        # egg-shaped ventricles along a tilted long axis (apex down, base up and to
        # one side); the closed ventricular cavity is removed — coronary blood
        # perfuses the muscle wall, not the chambers.
        d = np.array([0.45, 0.2, 0.87])
        d = d / np.linalg.norm(d)
        a = u * d[0] + v * d[1] + w * d[2]               # position along the long axis
        rho = np.sqrt(np.maximum(u**2 + v**2 + w**2 - a**2, 0))
        wall = (np.abs(a) <= 1) & (rho <= 0.8 * np.sqrt(np.clip(1 - a**2, 0, None))
                                   * (0.68 + 0.32 * a))
        ac = (a + 0.1) / 0.72                            # cavity: smaller, apex-ward, closed
        cavity = (np.abs(ac) < 1) & (rho <= 0.42 * np.sqrt(np.clip(1 - ac**2, 0, None))
                                     * (0.68 + 0.32 * ac))
        return wall & ~cavity
    if name == "kidney":                     # a bean: ellipsoid minus the hilum
        bean = (u / 0.55) ** 2 + (v / 0.42) ** 2 + w**2 <= 1
        hilum = (u + 0.62) ** 2 + v**2 + (w / 1.3) ** 2 <= 0.30**2
        return bean & ~hilum
    if name == "brain":                      # two hemispheres joined low down
        hemi = ((np.abs(u) - 0.42) / 0.55) ** 2 + (v / 0.95) ** 2 + (w / 0.7) ** 2 <= 1
        fissure = (np.abs(u) < 0.06) & (w > -0.05)     # longitudinal fissure, top half
        return hemi & ~fissure & (w > -0.6)
    if name == "liver":                      # large right lobe + smaller left lobe
        right = ((u + 0.25) / 0.75) ** 2 + (v / 0.62) ** 2 + ((w - 0.05) / 0.55) ** 2 <= 1
        left = ((u - 0.45) / 0.55) ** 2 + ((v - 0.05) / 0.45) ** 2 + ((w - 0.2) / 0.32) ** 2 <= 1
        return (right | left) & (w > -0.42 + 0.25 * u)
    if name == "lung":                       # a tall dome with a concave base
        dome = (u / 0.62) ** 2 + (v / 0.5) ** 2 + ((w + 0.55) / 1.5) ** 2 <= 1
        return dome & (w > -0.9 + 0.3 * (1 - (u / 0.62) ** 2 - (v / 0.5) ** 2))
    if name == "tumour":                     # a lobulated, seeded blob
        rng = np.random.default_rng(seed)
        r = np.sqrt(u**2 + v**2 + w**2) + 1e-12
        th, ph = np.arccos(np.clip(w / r, -1, 1)), np.arctan2(v, u)
        a, b, c = rng.uniform(0, 2 * np.pi, 3)
        rad = 0.78 * (1 + 0.16 * np.sin(3 * th + a) * np.sin(2 * ph + b)
                      + 0.10 * np.sin(5 * ph + c) * np.sin(th))
        return r <= rad
    raise ValueError(f"unknown 3-D shape {name!r}; choose from {list(SHAPES_3D)}")


def mask_3d(name: str, nx: int, ny: int, nz: int, domain, size: float = 0.85,
            seed: int = 0) -> np.ndarray:
    """The 3-D mask of ``name`` on the cell centres of an ``(nx, ny, nz)`` grid over
    ``domain = (x0, x1, y0, y1, z0, z1)``."""
    lo = np.asarray(domain[0::2], float)
    hi = np.asarray(domain[1::2], float)
    c, half = (lo + hi) / 2, size * (hi - lo) / 2
    if name not in STRETCH:
        half = np.full(3, half.min())
    axes = [lo[k] + (np.arange(m) + 0.5) * (hi[k] - lo[k]) / m for k, m in enumerate((nx, ny, nz))]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    return _unit_inside_3d(name, (X - c[0]) / half[0], (Y - c[1]) / half[1],
                           (Z - c[2]) / half[2], seed=seed)


# ── anatomical vessel entry points ────────────────────────────────────────────
# Where each organ's arterial inlet and venous outlet sit, in the same unit frame
# as its outline/mask: (position, direction the trunk grows into the tissue).
# 2-D directions are angles in degrees; 3-D are vectors. Roots sit at or just
# outside the surface (in the hilum / at the base), so the trunk enters the organ.
_ROOTS_2D = {
    #             artery: (x, y), angle           vein: (x, y), angle
    "ellipse":   (((0.0, -1.05), 90), ((0.0, 1.05), -90)),
    "rectangle": (((0.0, -1.05), 90), ((0.0, 1.05), -90)),
    "wedge":     (((0.0, -1.05), 90), ((0.0, 1.05), -90)),
    # coronary arteries leave the aortic root at the base and run to the apex;
    # the coronary sinus drains at the base into the right atrium
    "heart":     (((0.12, 0.74), -75), ((-0.62, 0.42), -40)),
    # renal artery and vein both at the hilum (the notch), entering medially
    "kidney":    (((-0.50, 0.10), 0), ((-0.50, -0.12), 0)),
    # internal carotid / circle of Willis from below; venous sinuses on top
    "brain":     (((0.05, -0.80), 90), ((0.35, 0.80), -100)),
    # hepatic artery (with the portal vein) at the porta hepatis on the inferior
    # surface; hepatic veins drain from the top into the IVC
    "liver":     (((0.05, -0.30), 90), ((-0.15, 0.95), -90)),
    # pulmonary artery and veins at the hilum on the medial side
    "lung":      (((-0.32, 0.02), 0), ((-0.30, -0.30), 0)),
    # lingual artery and vein enter at the root (base) of the tongue
    "tongue":    (((-0.22, -1.08), 90), ((0.22, -1.08), 90)),
    # umbilical vessels insert near the centre of the fetal (upper) surface
    "placenta":  (((-0.08, 0.52), -90), ((0.08, 0.52), -90)),
    # a feeding artery on one flank, the draining vein on the other
    "tumour":    (((-1.0, -0.15), 0), ((1.0, 0.15), 180)),
}

_HEART_AXIS = np.array([0.45, 0.2, 0.87]) / np.linalg.norm([0.45, 0.2, 0.87])
_HEART_SIDE = np.cross(_HEART_AXIS, [0.0, 1.0, 0.0])
_HEART_SIDE = _HEART_SIDE / np.linalg.norm(_HEART_SIDE)
_ROOTS_3D = {
    "ellipsoid":  (((0, 0, -1.05), (0, 0, 1)), ((0, 0, 1.05), (0, 0, -1))),
    "box":        (((0, 0, -1.05), (0, 0, 1)), ((0, 0, 1.05), (0, 0, -1))),
    "cylinder":   (((0, 0, -1.05), (0, 0, 1)), ((0, 0, 1.05), (0, 0, -1))),
    "hemisphere": (((0, 0, -1.05), (0, 0, 1)), ((0, 0, 1.05), (0, 0, -1))),
    # aortic root / coronary ostia at the base, coronary sinus beside it; both
    # trunks run from the base down the long axis toward the apex
    "heart": ((tuple(1.02 * _HEART_AXIS + 0.12 * _HEART_SIDE), tuple(-_HEART_AXIS)),
              (tuple(0.92 * _HEART_AXIS - 0.3 * _HEART_SIDE), tuple(-_HEART_AXIS))),
    "kidney":     (((-0.6, 0, 0.1), (1, 0, 0)), ((-0.6, 0, -0.12), (1, 0, 0))),
    # circle of Willis below; superior sagittal sinus on top, its trunk entering a
    # hemisphere (exactly on the midline it would run down the empty fissure)
    "brain":      (((0, 0.1, -0.68), (0, 0, 1)), ((0.16, -0.3, 0.74), (0.3, 0.1, -1))),
    "liver":      (((0.05, 0, -0.5), (0, 0, 1)), ((-0.1, 0.55, 0.55), (0, -0.5, -0.6))),
    "lung":       (((-0.68, 0, -0.2), (1, 0, 0)), ((-0.68, 0.1, -0.42), (1, 0, 0))),
    "tumour":     (((-0.95, 0, 0), (1, 0, 0)), ((0.95, 0, 0), (-1, 0, 0))),
}


def _frame(name, domain, size):
    """Centre + half-extents used to place unit coordinates (as polygon/mask do)."""
    lo, hi = np.asarray(domain[0::2], float), np.asarray(domain[1::2], float)
    c, half = (lo + hi) / 2, size * (hi - lo) / 2
    if name not in STRETCH:
        half = np.full(len(lo), half.min())
    return c, half


def roots(name: str, dim: int, domain, size: float = 0.85) -> dict:
    """Anatomical artery/vein roots of ``name`` placed in ``domain``.

    Returns ``{"artery": {"root": [...], "angle"|"direction": ...}, "vein": {...}}``
    — a 2-D ``angle`` in degrees, or a 3-D unit ``direction`` vector — matching
    where the organ's arterial inlet and venous outlet sit anatomically.
    """
    table = _ROOTS_3D if dim == 3 else _ROOTS_2D
    if name not in table:
        raise ValueError(f"no anatomical roots for {name!r} in {dim}-D")
    c, half = _frame(name, domain, size)
    out = {}
    for kind, (pos, aim) in zip(("artery", "vein"), table[name], strict=True):
        root = [float(v) for v in c + half * np.asarray(pos, float)]
        if dim == 3:
            d = half * np.asarray(aim, float)            # follow any stretch of the frame
            out[kind] = {"root": root, "direction": [float(v) for v in d / np.linalg.norm(d)]}
        else:
            a = np.radians(aim)                          # aim in unit space, then stretch
            d = half * np.array([np.cos(a), np.sin(a)])
            out[kind] = {"root": root, "angle": float(np.degrees(np.arctan2(d[1], d[0])))}
    return out

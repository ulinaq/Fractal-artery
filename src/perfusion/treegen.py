"""Tree generation for the Tree Explorer (service layer): grow an arterial and a
venous tree into a chosen tissue shape, in 2-D or 3-D, and optionally solve.

A tree is built in two stages, each user-controllable:

1. **Fractal skeleton** — a self-similar bifurcating tree from a placed root
   (position + direction) with ``generations`` levels of branching
   (:func:`perfusion.core.geometry.grow_tree` / ``grow_tree_3d``). Each extra
   generation adds a level of branches.
2. **Space filling** (optional) — the skeleton's tips are grown on so the tree
   fills the tissue: ``"colonization"`` (space colonisation, ``fill_steps``
   growth iterations toward attraction points sampled in the tissue) or
   ``"cco"`` (constrained constructive optimisation, ``n_terminals`` terminals
   added one at a time). Both work in 2-D and 3-D.

The tissue is a preset shape (or a custom 2-D polygon) rasterised to the Darcy
grid's mask, so growth *and* the perfusion solve are organ-shaped.

Depends only on NumPy/SciPy, ``core`` and :mod:`perfusion.growth` — no web deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import growth
from .core.darcy import Grid
from .core.geometry import Network, grow_tree, grow_tree_3d, rasterize_mask
from .core.parameters import Parameters, bifurcation_geometry

DOMAIN_2D = (-70.0, 70.0, 0.0, 160.0)
DOMAIN_3D = (0.0, 60.0, 0.0, 60.0, 0.0, 60.0)
SHAPES_2D = ("rectangle", "ellipse", "kidney", "wedge", "custom")
SHAPES_3D = ("box", "ellipsoid", "cylinder", "hemisphere")
FILLS = ("none", "colonization", "cco")


@dataclass
class TreeSpec:
    """One tree's initial condition and branching law."""

    root: tuple = (0.0, 0.0)              # (x, y) or (x, y, z)
    angle: float = 90.0                   # 2-D trunk direction, degrees from +x
    direction: tuple = (0.0, 0.0, 1.0)    # 3-D trunk direction
    radius: float = 2.0                   # trunk radius
    gamma: float = 1.4                    # branching asymmetry (>= 1; 1 = symmetric)
    branch_angle: float = 35.0            # 3-D half-angle between children, degrees


@dataclass
class GenSpec:
    """Everything the explorer can vary."""

    dim: int = 2
    shape: str = "ellipse"
    polygon: list | None = None           # custom 2-D tissue outline (domain coords)
    size: float = 0.85                    # preset shape scale (fraction of the domain)
    artery: TreeSpec = field(default_factory=lambda: TreeSpec((0.0, 8.0), 90.0))
    vein: TreeSpec = field(default_factory=lambda: TreeSpec((0.0, 152.0), -90.0, radius=2.4,
                                                            gamma=1.2))
    generations: int = 4                  # fractal levels of branching
    length_to_radius: float = 10.0
    exponent: float = 2.8                 # Murray bifurcation exponent
    fill: str = "colonization"            # "none" | "colonization" | "cco"
    fill_target: str = "both"             # "both" | "artery" | "vein"
    fill_steps: int = 40                  # colonisation iterations ("growth steps")
    n_points: int = 400                   # colonisation attraction points
    step: float | None = None             # colonisation segment length (None = auto)
    influence: float | None = None        # colonisation influence radius (None = auto)
    n_terminals: int = 60                 # CCO terminals per tree
    leaf_radius: float | None = None      # Murray leaf radius for filled trees
    seed: int = 0
    nx: int = 60                          # Darcy grid (2-D); 3-D uses nx for each axis
    ny: int = 70


# ── tissue shapes ─────────────────────────────────────────────────────────────
def shape_polygon(shape: str, size: float = 0.85, domain=DOMAIN_2D, n: int = 24) -> list:
    """A preset 2-D tissue outline (closed polygon, domain coordinates). ``n``
    vertices for curved shapes — few enough to drag as handles in the explorer."""
    x0, x1, y0, y1 = domain
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax, ay = size * (x1 - x0) / 2, size * (y1 - y0) / 2
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    if shape == "rectangle":
        pts = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
        return [[cx + ax * u, cy + ay * v] for u, v in pts]
    if shape == "wedge":                     # a fan opening upward from the base
        pts = [(-0.25, -1), (0.25, -1), (1, 1), (-1, 1)]
        return [[cx + ax * u, cy + ay * v] for u, v in pts]
    if shape == "kidney":                    # an ellipse with a hilum notch on the left
        r = 1 - 0.32 * np.exp(-((np.angle(np.exp(1j * (t - np.pi)))) ** 2) / 0.18)
        return [[cx + ax * r[i] * np.cos(t[i]), cy + ay * r[i] * np.sin(t[i])] for i in range(n)]
    if shape == "ellipse":
        return [[cx + ax * np.cos(a), cy + ay * np.sin(a)] for a in t]
    raise ValueError(f"unknown 2-D shape {shape!r}; choose from {SHAPES_2D}")


def shape_mask_3d(shape: str, nx: int, ny: int, nz: int, size: float = 0.85,
                  domain=DOMAIN_3D) -> np.ndarray:
    """A preset 3-D tissue mask on the cell centres of an ``(nx, ny, nz)`` grid."""
    lo = np.asarray(domain[0::2], float)
    hi = np.asarray(domain[1::2], float)
    c, half = (lo + hi) / 2, size * (hi - lo) / 2
    axes = [lo[k] + (np.arange(m) + 0.5) * (hi[k] - lo[k]) / m for k, m in enumerate((nx, ny, nz))]
    X, Y, Z = np.meshgrid(*axes, indexing="ij")
    u, v, w = (X - c[0]) / half[0], (Y - c[1]) / half[1], (Z - c[2]) / half[2]
    if shape == "box":
        return (np.abs(u) <= 1) & (np.abs(v) <= 1) & (np.abs(w) <= 1)
    if shape == "ellipsoid":
        return u**2 + v**2 + w**2 <= 1
    if shape == "cylinder":                  # axis along z
        return (u**2 + v**2 <= 1) & (np.abs(w) <= 1)
    if shape == "hemisphere":                # dome sitting on the bottom of the box
        wz = (Z - lo[2]) / (2 * half[2])
        return u**2 + v**2 + wz**2 <= 1
    raise ValueError(f"unknown 3-D shape {shape!r}; choose from {SHAPES_3D}")


def tissue(spec: GenSpec):
    """``(mask, domain, outline)``: the tissue mask on the Darcy grid, the domain
    window, and the 2-D outline polygon (``None`` in 3-D)."""
    if spec.dim == 3:
        n = int(spec.nx)
        return shape_mask_3d(spec.shape, n, n, n, spec.size), DOMAIN_3D, None
    poly = spec.polygon if spec.shape == "custom" else shape_polygon(spec.shape, spec.size)
    if not poly or len(poly) < 3:
        raise ValueError("a custom tissue shape needs a polygon with at least 3 points")
    mask = rasterize_mask(spec.nx, spec.ny, DOMAIN_2D, [poly])
    if not mask.any():
        raise ValueError("the tissue shape covers no grid cells")
    return mask, DOMAIN_2D, [list(map(float, p)) for p in poly]


# ── trees ─────────────────────────────────────────────────────────────────────
def _branching_tables(gamma: float, exponent: float, generations: int):
    phi, ratio = bifurcation_geometry(gamma, exponent)
    g = max(int(generations), 1)
    return np.tile(phi, (g, 1)), np.tile(ratio, (g, 1))


def fractal_tree(t: TreeSpec, spec: GenSpec, *, venous: bool) -> Network:
    """The fractal skeleton: ``spec.generations`` levels of branching from ``t.root``."""
    g = int(spec.generations)
    if spec.dim == 3:
        ratio = 2.0 ** (-1.0 / spec.exponent)            # symmetric Murray split
        return grow_tree_3d(tuple(t.root), tuple(t.direction), root_radius=t.radius,
                            min_radius=0.0, length_to_radius=spec.length_to_radius,
                            theta=np.radians(t.branch_angle), ratio=ratio,
                            max_generations=g, mirror=venous)
    angles, ratios = _branching_tables(t.gamma, spec.exponent, g)
    return grow_tree(tuple(t.root[:2]), np.radians(t.angle), t.radius, 0.0,
                     spec.length_to_radius, angles, ratios,
                     mirror_children=not venous, max_generations=g)


def fill_tree(net: Network, mask, domain, spec: GenSpec, seed: int) -> Network:
    """Grow ``net``'s tips on into the tissue (colonisation / CCO), or pass through."""
    if spec.fill == "none":
        return net
    leaf = spec.leaf_radius if spec.leaf_radius else float(net.seg_radius.min())
    if spec.fill == "cco":
        pool = growth.sample_points_in_mask(mask, domain, 6 * spec.n_terminals, seed=seed)
        return growth.cco_grow(net, pool, n_terminals=spec.n_terminals, seed=seed,
                               exponent=spec.exponent, leaf_radius=leaf)
    if spec.fill == "colonization":
        pts = growth.sample_points_in_mask(mask, domain, spec.n_points, seed=seed)
        return growth.grow_network(net, pts, step=spec.step, influence_radius=spec.influence,
                                   max_iter=int(spec.fill_steps), exponent=spec.exponent,
                                   leaf_radius=leaf)
    raise ValueError(f"unknown fill {spec.fill!r}; choose from {FILLS}")


def build(spec: GenSpec):
    """Build both trees in the tissue. Returns ``(art, ven, mask, domain, outline)``."""
    if spec.dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")
    mask, domain, outline = tissue(spec)
    art = fractal_tree(spec.artery, spec, venous=False)
    ven = fractal_tree(spec.vein, spec, venous=True)
    if spec.fill_target in ("both", "artery"):
        art = fill_tree(art, mask, domain, spec, spec.seed)
    if spec.fill_target in ("both", "vein"):
        ven = fill_tree(ven, mask, domain, spec, spec.seed + 1)
    return art, ven, mask, domain, outline


def tree_stats(net: Network) -> dict:
    from .core.morphometry import strahler_summary

    return {"segments": int(net.n_segments), "terminals": int(net.terminal_nodes.size),
            "total_length": float(net.seg_length.sum()),
            "max_order": int(strahler_summary(net)["max_order"]) if net.n_segments else 0}


def solve(spec: GenSpec, art: Network, ven: Network, mask, domain,
          p_in: float = 10.0, p_out: float = 0.0):
    """Solve the hybrid perfusion model on the tissue (2-D or 3-D)."""
    prm = Parameters(p_in=p_in, p_out=p_out)
    if spec.dim == 3:
        from .core.darcy3d import Grid3D
        from .core.pipeline3d import run_hybrid_3d

        lo, hi = np.asarray(domain[0::2]), np.asarray(domain[1::2])
        n = mask.shape[0]
        grid = Grid3D(nx=n, ny=n, nz=n, lx=hi[0] - lo[0], ly=hi[1] - lo[1], lz=hi[2] - lo[2],
                      mask=mask)
        return run_hybrid_3d(prm, grid, art, ven, domain, eps=2.0)
    from .core.pipeline import run_hybrid

    grid = Grid(nx=spec.nx, ny=spec.ny, lx=domain[1] - domain[0], ly=domain[3] - domain[2],
                mask=mask)
    return run_hybrid(prm, grid, domain, art=art, ven=ven)

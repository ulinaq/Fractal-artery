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

from . import growth, organs
from .core.darcy import Grid
from .core.geometry import Network, grow_tree, grow_tree_3d, rasterize_mask
from .core.parameters import Parameters, bifurcation_geometry

DOMAIN_2D = (-70.0, 70.0, 0.0, 160.0)          # at extent 1
DOMAIN_3D = (0.0, 60.0, 0.0, 60.0, 0.0, 60.0)
SHAPES_2D = (*organs.SHAPES_2D, "custom")
SHAPES_3D = tuple(organs.SHAPES_3D)
FILLS = ("none", "colonization", "cco")


def domain_for(dim: int, extent: float = 1.0) -> tuple:
    """The physical window, scaled by ``extent`` (bigger objects): the 2-D window
    grows about its bottom-centre, the 3-D box from its origin corner."""
    e = float(extent)
    if dim == 3:
        return tuple(v * e for v in DOMAIN_3D)
    return (DOMAIN_2D[0] * e, DOMAIN_2D[1] * e, 0.0, DOMAIN_2D[3] * e)


@dataclass
class TreeSpec:
    """One tree's initial condition and branching law."""

    root: tuple = (0.0, 0.0)              # (x, y) or (x, y, z)
    angle: float = 90.0                   # 2-D trunk direction, degrees from +x
    direction: tuple = (0.0, 0.0, 1.0)    # 3-D trunk direction
    radius: float = 2.0                   # trunk radius
    gamma: float = 1.4                    # branching asymmetry (>= 1; 1 = symmetric)


@dataclass
class GenSpec:
    """Everything the explorer can vary."""

    dim: int = 2
    shape: str = "ellipse"
    polygon: list | None = None           # custom 2-D tissue outline (domain coords)
    size: float = 0.85                    # preset shape scale (fraction of the domain)
    extent: float = 1.0                   # domain scale: >1 explores a bigger object
    artery: TreeSpec = field(default_factory=lambda: TreeSpec((0.0, 8.0), 90.0))
    vein: TreeSpec = field(default_factory=lambda: TreeSpec((0.0, 152.0), -90.0, radius=2.4,
                                                            gamma=1.2))
    generations: int = 4                  # fractal levels of branching
    length_to_radius: float = 10.0
    exponent: float = 2.8                 # Murray bifurcation exponent
    twist: float = 90.0                   # 3-D: rotation of each branching plane (deg)
    jitter: float = 0.3                   # 3-D: random variation of angles/lengths (0..1)
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
def shape_polygon(shape: str, size: float = 0.85, domain=DOMAIN_2D, seed: int = 0) -> list:
    """A preset 2-D tissue outline (closed polygon, domain coordinates) from the
    organ library (:mod:`perfusion.organs`)."""
    return organs.polygon(shape, domain, size, seed=seed)


def shape_mask_3d(shape: str, nx: int, ny: int, nz: int, size: float = 0.85,
                  domain=DOMAIN_3D, seed: int = 0) -> np.ndarray:
    """A preset 3-D tissue mask on the cell centres of an ``(nx, ny, nz)`` grid."""
    return organs.mask_3d(shape, nx, ny, nz, domain, size, seed=seed)


def tissue(spec: GenSpec):
    """``(mask, domain, outline)``: the tissue mask on the Darcy grid, the domain
    window, and the 2-D outline polygon (``None`` in 3-D)."""
    domain = domain_for(spec.dim, spec.extent)
    if spec.dim == 3:
        n = int(spec.nx)
        mask = shape_mask_3d(spec.shape, n, n, n, spec.size, domain, seed=spec.seed)
        if not mask.any():
            raise ValueError("the tissue shape covers no grid cells — raise the resolution")
        return mask, domain, None
    poly = (spec.polygon if spec.shape == "custom"
            else shape_polygon(spec.shape, spec.size, domain, seed=spec.seed))
    if not poly or len(poly) < 3:
        raise ValueError("a custom tissue shape needs a polygon with at least 3 points")
    mask = rasterize_mask(spec.nx, spec.ny, domain, [poly])
    if not mask.any():
        raise ValueError("the tissue shape covers no grid cells")
    return mask, domain, [list(map(float, p)) for p in poly]


# ── trees ─────────────────────────────────────────────────────────────────────
def _branching_tables(gamma: float, exponent: float, generations: int):
    phi, ratio = bifurcation_geometry(gamma, exponent)
    g = max(int(generations), 1)
    return np.tile(phi, (g, 1)), np.tile(ratio, (g, 1))


def fractal_tree(t: TreeSpec, spec: GenSpec, *, venous: bool) -> Network:
    """The fractal skeleton: ``spec.generations`` levels of branching from ``t.root``."""
    g = int(spec.generations)
    if spec.dim == 3:      # truly 3-D: twisting branch planes, Murray-law asymmetry
        return grow_tree_3d(tuple(t.root), tuple(t.direction), root_radius=t.radius,
                            min_radius=0.0, length_to_radius=spec.length_to_radius,
                            max_generations=g, mirror=venous, gamma=t.gamma,
                            exponent=spec.exponent, twist=np.radians(spec.twist),
                            jitter=spec.jitter, seed=spec.seed + (7 if venous else 0))
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


def homogeneity(res) -> dict:
    """Perfusion-homogeneity measures of a solved 2-D or 3-D result (Paper D):
    flux range/std/L2/mean and time-of-flight std/max over the active tissue."""
    mask = np.asarray(res.grid.mask, bool)
    v = res.flux[mask]
    v = v[np.isfinite(v)]
    tof = res.tof[mask]
    tof = tof[np.isfinite(tof) & (tof > 0)]
    out = {"flux_range": float(v.max() - v.min()) if v.size else float("inf"),
           "flux_std": float(v.std()) if v.size else float("inf"),
           "flux_l2": float(np.linalg.norm(v)), "flux_mean": float(v.mean()) if v.size else 0.0,
           "tof_std": float(tof.std()) if tof.size else float("inf"),
           "tof_max": float(tof.max()) if tof.size else float("inf")}
    return out

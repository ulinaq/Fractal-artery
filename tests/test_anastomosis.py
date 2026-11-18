"""Anastomoses — loops added to a tree, and the collateral flow they carry."""

import numpy as np
import pytest

from perfusion import Grid, Parameters
from perfusion.core import anastomosis as an
from perfusion.core.geometry import arterial_tree, grow_tree_3d, venous_tree
from perfusion.core.pipeline import run_hybrid

DOM = (-70, 70, 0, 160)


@pytest.fixture(scope="module")
def trees():
    prm = Parameters(r_min_frac=0.3, seed=0)
    return prm, arterial_tree(prm), venous_tree(prm)


def _territory_flux(prm, art, ven, terr, grid):
    r = run_hybrid(prm, grid, DOM, art=art, ven=ven)
    tf = r.solution.terminal_flux[:art.terminal_nodes.size]
    idx = {int(n): i for i, n in enumerate(art.terminal_nodes)}
    return sum(tf[idx[int(n)]] for n in terr), r


def test_links_make_loops_and_keep_the_tree_intact(trees):
    _, art, _ = trees
    assert an.n_loops(art) == 0
    looped, links = an.add_anastomoses(art, n_links=5)
    assert len(links) == 5
    assert an.n_loops(looped) == 5                       # one independent loop per link
    np.testing.assert_array_equal(looped.terminal_nodes, art.terminal_nodes)
    np.testing.assert_array_equal(looped.node_x[:art.n_nodes], art.node_x)   # indices kept
    for link in links:
        s1, s2 = link["segments"]
        assert link["radius"] == pytest.approx(0.7 * min(art.seg_radius[s1], art.seg_radius[s2]))
        a, b = looped.positions([link["a"], link["b"]])
        assert link["length"] == pytest.approx(np.linalg.norm(a - b))
        # never siblings: their nearest endpoints are >= 2 junctions apart
        assert not ({int(art.seg_start[s1]), int(art.seg_end[s1])}
                    & {int(art.seg_start[s2]), int(art.seg_end[s2])})
    # total vessel length grows only by the links (splitting conserves length)
    assert looped.seg_length.sum() == pytest.approx(
        art.seg_length.sum() + sum(link["length"] for link in links))


def test_loops_conserve_mass_and_rescue_an_occluded_territory(trees):
    """Blocking a branch starves its territory in a tree; with collaterals some flow
    reaches it the other way round — more with wider links (Poiseuille ∝ r⁴)."""
    prm, art, ven = trees
    grid = Grid(nx=60, ny=70, lx=140, ly=160)
    thin, links = an.add_anastomoses(art, n_links=8, radius_frac=0.5)
    wide, _ = an.add_anastomoses(art, n_links=8, radius_frac=1.0)
    linked = {s for link in links for s in link["segments"]}
    rescued = []
    for s in range(art.n_segments):
        terr = an.downstream_terminals(art, s)
        if not 3 <= terr.size <= 9 or s in linked:
            continue
        q0, _ = _territory_flux(prm, art, ven, terr, grid)
        q_tree, _ = _territory_flux(prm, an.occlude(art, s), ven, terr, grid)
        q_thin, _ = _territory_flux(prm, an.occlude(thin, s), ven, terr, grid)
        q_wide, r = _territory_flux(prm, an.occlude(wide, s), ven, terr, grid)
        assert q_tree < 1e-6 * q0                             # a tree: the territory starves
        tf = r.solution.terminal_flux
        assert tf.sum() == pytest.approx(0.0, abs=1e-12 * np.abs(tf).sum())   # in == out
        if q_wide > 0.05 * q0:
            assert q_thin < q_wide
            rescued.append(q_wide / q0)
    assert len(rescued) >= 3


def test_anastomoses_in_3d():
    art = grow_tree_3d((0.0, 0.0, 0.0), (0, 0, 1), root_radius=2.0, min_radius=0.4,
                       length_to_radius=6.0, max_generations=6, seed=1)
    looped, links = an.add_anastomoses(art, n_links=4)
    assert links
    assert looped.is_3d
    assert an.n_loops(looped) == len(links)
    a, b = looped.positions([links[0]["a"], links[0]["b"]])
    assert a.shape == (3,)
    assert links[0]["length"] == pytest.approx(np.linalg.norm(a - b))


def test_downstream_terminals_and_validation(trees):
    _, art, _ = trees
    trunk = int(np.flatnonzero(np.isin(art.seg_start, art.inlet_nodes)
                               | np.isin(art.seg_end, art.inlet_nodes))[0])
    np.testing.assert_array_equal(np.sort(an.downstream_terminals(art, trunk)),
                                  np.sort(art.terminal_nodes))      # the trunk feeds all
    leaf_seg = int(np.flatnonzero(np.isin(art.seg_end, art.terminal_nodes))[0])
    assert an.downstream_terminals(art, leaf_seg).size == 1
    assert an.add_anastomoses(art, n_links=0) == (art, [])
    with pytest.raises(ValueError, match="min_detour"):
        an.add_anastomoses(art, min_detour=1.0)
    with pytest.raises(ValueError, match="radius_frac"):
        an.add_anastomoses(art, radius_frac=0.0)


def test_other_analyses_tolerate_loops(trees):
    """Pruning, Strahler morphometry, the nonlinear solve and the direct model run on a
    looped tree."""
    from perfusion.core import morphometry
    from perfusion.core.direct import connect_terminals, solve_direct
    from perfusion.core.geometry import prune_terminal_branches

    prm, art, ven = trees
    looped, _ = an.add_anastomoses(art, n_links=4)
    pruned = prune_terminal_branches(looped, 1)
    assert pruned.n_segments < looped.n_segments
    morphometry.strahler_orders(looped)
    grid = Grid(nx=40, ny=46, lx=140, ly=160)
    nl = run_hybrid(prm, grid, DOM, art=looped, ven=ven, nonlinear=True)
    assert np.all(np.isfinite(nl.solution.q_a))
    sol = solve_direct(looped, ven, connect_terminals(looped, ven, rho_c=0.0), 10.0, 0.0)
    assert sol.q_a[0] == pytest.approx(sol.q_c.sum(), rel=1e-6)

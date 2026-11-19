"""Tests for the hybrid vascular-network + Darcy solver."""

import numpy as np
import pytest

from perfusion import darcy, network
from perfusion.core.network import _Layout


def _response(grid, art, ven, domain_limit=(-70, 70, 0, 160), eps=4.0):
    from perfusion.core.pipeline import _terminal_cells

    cells = _terminal_cells(grid, art, ven, domain_limit)
    return darcy.terminal_response(grid, cells, eps)


def test_layout_offsets(art, ven):
    lay = _Layout(art, ven)
    assert lay.pa == 0
    assert lay.qa == art.n_nodes
    assert lay.size == art.n_nodes + art.n_segments + ven.n_nodes + ven.n_segments + 1
    cols, signs, ta, _ = lay.terminal_flux_columns()
    assert cols.shape == signs.shape
    assert np.all(signs[:len(ta)] == 1)
    assert np.all(signs[len(ta):] == -1)


def test_multi_component_trees_conserve_mass(grid):
    # two separate arterial feeding trees + two venous trees share one Darcy bed
    from perfusion.core.geometry import network_from_polylines

    art = network_from_polylines([
        [(-40, 5), (-40, 40)], [(-40, 40), (-55, 80)], [(-40, 40), (-25, 80)],
        [(40, 5), (40, 40)], [(40, 40), (25, 80)], [(40, 40), (55, 80)],
    ], radii=2.0)
    ven = network_from_polylines([
        [(-40, 155), (-40, 110)], [(-40, 110), (-55, 80)], [(-40, 110), (-25, 80)],
        [(40, 155), (40, 110)], [(40, 110), (25, 80)], [(40, 110), (55, 80)],
    ], radii=2.0)
    assert len(art.inlet_nodes) == 2       # one inlet per arterial feeding tree
    assert len(ven.inlet_nodes) == 2       # one outlet per venous tree

    response, _, _ = _response(grid, art, ven)
    sol = network.solve_linear(art, ven, response, p_in=10.0, p_out=0.0)
    assert np.allclose(sol.p_a[art.inlet_nodes], 10.0)      # both feeding trees driven
    assert np.allclose(sol.p_v[ven.inlet_nodes], 0.0)
    assert np.isclose(sol.terminal_flux.sum(), 0.0, atol=1e-8)   # global mass balance


def test_multiple_inlets_outlets_conserve_mass(art, ven, grid):
    import dataclasses

    ta, tv = art.terminal_nodes, ven.terminal_nodes
    # promote extra leaves to boundary nodes: 3 arterial inlets, 2 venous outlets
    art2 = dataclasses.replace(art, inlet_nodes=np.array([art.root_node, int(ta[0]), int(ta[1])]))
    ven2 = dataclasses.replace(ven, inlet_nodes=np.array([ven.root_node, int(tv[0])]))
    response, _, _ = _response(grid, art2, ven2)
    sol = network.solve_linear(art2, ven2, response, p_in=10.0, p_out=0.0)

    assert np.allclose(sol.p_a[art2.inlet_nodes], 10.0)     # every inlet at p_in
    assert np.allclose(sol.p_v[ven2.inlet_nodes], 0.0)      # every outlet at p_out
    assert np.isclose(sol.terminal_flux.sum(), 0.0, atol=1e-8)
    # promoted leaves are no longer Darcy terminals
    assert set(art2.terminal_nodes.tolist()).isdisjoint(art2.inlet_nodes.tolist())


def test_linear_solve_conserves_mass(art, ven, grid):
    response, _, _ = _response(grid, art, ven)
    sol = network.solve_linear(art, ven, response, p_in=10.0, p_out=0.0)
    # arterial trunk inflow == sum of arterial terminal outflows
    term = sol.q_a[art.incoming_segment[art.terminal_nodes]].sum()
    assert sol.q_a[0] == pytest.approx(term, rel=1e-6)
    # global balance: arterial inflow == venous outflow
    assert sol.q_a[0] == pytest.approx(sol.q_v[0], rel=1e-6)
    assert sol.q_a[0] > 0                       # positive driving pressure


def test_pressure_decreases_from_artery_to_vein(art, ven, grid):
    response, _, _ = _response(grid, art, ven)
    sol = network.solve_linear(art, ven, response, p_in=10.0, p_out=0.0)
    assert sol.p_a[art.root_node] == pytest.approx(10.0)
    assert sol.p_v[ven.root_node] == pytest.approx(0.0)
    # arterial terminals sit above venous terminals in pressure
    assert sol.p_a[art.terminal_nodes].mean() > sol.p_v[ven.terminal_nodes].mean()


def test_zero_driving_pressure_gives_no_flow(art, ven, grid):
    response, _, _ = _response(grid, art, ven)
    sol = network.solve_linear(art, ven, response, p_in=5.0, p_out=5.0)
    assert np.allclose(sol.q_a, 0.0, atol=1e-9)
    assert np.allclose(sol.terminal_flux, 0.0, atol=1e-9)


def test_nonlinear_matches_linear_without_losses(art, ven, grid, small_params):
    response, _, _ = _response(grid, art, ven)
    lin = network.solve_linear(art, ven, response, 10.0, 0.0)
    non = network.solve_nonlinear(art, ven, response, 10.0, 0.0, small_params,
                                  junction_loss=False)
    # with junction losses disabled the nonlinear system reduces to the linear one
    assert np.allclose(non.q_a, lin.q_a, atol=1e-6)


def test_nonlinear_conserves_mass(art, ven, grid, small_params):
    response, _, _ = _response(grid, art, ven)
    sol = network.solve_nonlinear(art, ven, response, 10.0, 0.0, small_params,
                                  junction_loss=True)
    assert sol.q_a[0] == pytest.approx(sol.q_v[0], rel=1e-5)


def test_nonlinear_junction_loss_reduces_flow(art, ven, grid, small_params):
    # Adding bifurcation pressure losses raises resistance -> less trunk flow.
    response, _, _ = _response(grid, art, ven)
    lin = network.solve_linear(art, ven, response, 10.0, 0.0)
    non = network.solve_nonlinear(art, ven, response, 10.0, 0.0, small_params,
                                  junction_loss=True)
    assert non.q_a[0] < lin.q_a[0]


def test_per_tree_viscosity(art, ven, grid):
    # A (mu_art, mu_ven) pair must match passing the same scalar to both trees.
    response, _, _ = _response(grid, art, ven)
    scalar = network.solve_linear(art, ven, response, 10.0, 0.0, 2.0)
    paired = network.solve_linear(art, ven, response, 10.0, 0.0, (2.0, 2.0))
    assert np.allclose(scalar.q_a, paired.q_a)
    # per-segment arrays are accepted too
    mu_a = np.full(art.n_segments, 2.0)
    arr = network.solve_linear(art, ven, response, 10.0, 0.0, (mu_a, 2.0))
    assert np.allclose(scalar.q_a, arr.q_a)


# ── monolithic (sparse) coupling == Schur (dense response) ─────────────────────
@pytest.mark.parametrize("kw", [{}, {"nonlinear": True}, {"elasticity": True},
                                {"phase_separation": True}])
def test_monolithic_coupling_matches_schur(kw):
    from perfusion import Grid, Parameters, run_hybrid
    prm = Parameters(r_min_frac=0.3, compliance=0.02, fahraeus_lindqvist=True)
    grid = Grid(nx=50, ny=58, lx=140, ly=160)
    a = run_hybrid(prm, grid, coupling="schur", **kw)
    b = run_hybrid(prm, grid, coupling="monolithic", **kw)
    scale = np.nanmax(np.abs(a.pressure))
    assert np.nanmax(np.abs(a.pressure - b.pressure)) < 1e-9 * scale
    assert np.allclose(a.solution.terminal_flux, b.solution.terminal_flux,
                       rtol=1e-9, atol=1e-13)
    assert np.allclose(a.solution.q_a, b.solution.q_a, rtol=1e-9, atol=1e-13)
    assert b.solution.darcy_pressure is not None          # the bed came from the solve
    q = b.solution.terminal_flux
    assert q[q > 0].sum() == pytest.approx(-q[q < 0].sum(), rel=1e-9)


def test_monolithic_system_is_sparse():
    """No dense (n_cells × T) block: the assembled matrix has O(nnz) entries."""
    from perfusion import Grid, Parameters
    from perfusion.core import darcy
    from perfusion.core.geometry import arterial_tree, venous_tree
    from perfusion.core.network import _assemble, _resistances
    from perfusion.core.pipeline import _terminal_cells
    prm, grid = Parameters(r_min_frac=0.2), Grid(nx=60, ny=70, lx=140, ly=160)
    art, ven = arterial_tree(prm), venous_tree(prm)
    cells = _terminal_cells(grid, art, ven, (-70, 70, 0, 160))
    op, _ = darcy.coupling_operator(grid, cells, 4.0)
    sysm = _assemble(art, ven, op, 1.0, 0.0, *_resistances(art, ven, 1.0))
    n, t = grid.n_cells, len(cells)
    assert sysm.A.nnz < 20 * (n + art.n_nodes + ven.n_nodes)      # O(cells + network)
    assert sysm.A.nnz < n * t / 10                                # ≪ the dense n×T block


def test_coupling_choice_is_validated_and_auto_switches():
    from perfusion.core.pipeline import MONOLITHIC_AUTO_ENTRIES, use_monolithic
    assert not use_monolithic("auto", 1000, 100)
    assert use_monolithic("auto", MONOLITHIC_AUTO_ENTRIES, 2)
    assert use_monolithic("monolithic", 10, 1)
    assert not use_monolithic("schur", 10**9, 10**4)
    with pytest.raises(ValueError, match="coupling must be"):
        use_monolithic("nope", 1, 1)


def _reverse_segment(net, s):
    """The same tree with segment ``s`` drawn the other way round."""
    import dataclasses

    ss, se = net.seg_start.copy(), net.seg_end.copy()
    ss[s], se[s] = net.seg_end[s], net.seg_start[s]
    return dataclasses.replace(net, seg_start=ss, seg_end=se)


def test_reversed_leaf_segment_gives_the_same_flow(art, ven, grid):
    # A hand-drawn vessel may point either way; the terminal flux must follow the
    # topology, not the drawing direction (it used incoming_segment and a fixed sign).
    response, _, _ = _response(grid, art, ven)
    ref = network.solve_linear(art, ven, response, p_in=10.0, p_out=0.0)
    sa = int(art.leaf_segments(art.terminal_nodes[:1])[0][0])
    sv = int(ven.leaf_segments(ven.terminal_nodes[:1])[0][0])
    art_r, ven_r = _reverse_segment(art, sa), _reverse_segment(ven, sv)
    sol = network.solve_linear(art_r, ven_r, response, p_in=10.0, p_out=0.0)
    assert np.allclose(sol.terminal_flux, ref.terminal_flux, rtol=1e-9, atol=1e-14)
    assert np.allclose(sol.p_a, ref.p_a)
    assert np.allclose(sol.p_v, ref.p_v)
    assert sol.q_a[sa] == pytest.approx(-ref.q_a[sa])      # only the sign convention flips
    assert sol.q_v[sv] == pytest.approx(-ref.q_v[sv])


def test_root_leaf_becomes_a_proper_terminal(art, ven, grid):
    # Moving the inlet to another leaf turns the BFS root into a Darcy terminal; it
    # has no incoming segment, and its flux used to read a *pressure* unknown.
    import dataclasses

    leaf = int(art.terminal_nodes[0])
    art2 = dataclasses.replace(art, inlet_nodes=np.array([leaf]))
    assert art.root_node in art2.terminal_nodes
    response, _, _ = _response(grid, art2, ven)
    sol = network.solve_linear(art2, ven, response, p_in=10.0, p_out=0.0)
    n_art = art2.terminal_nodes.size
    inflow = sol.terminal_flux[:n_art].sum()
    assert inflow > 0
    assert inflow == pytest.approx(-sol.terminal_flux[n_art:].sum(), rel=1e-8)
    seg, orient = art2.leaf_segments([leaf])
    assert orient[0] * sol.q_a[seg[0]] == pytest.approx(-inflow, rel=1e-8)  # all leaves the inlet


def test_leaf_segments_rejects_isolated_nodes(art):
    import dataclasses

    lone = dataclasses.replace(art, node_x=np.append(art.node_x, 0.0),
                               node_y=np.append(art.node_y, 0.0),
                               node_role=np.append(art.node_role, 0))
    with pytest.raises(ValueError, match="no incident segment"):
        lone.leaf_segments([lone.n_nodes - 1])


def test_3d_junction_loss_uses_the_turning_angle():
    from perfusion.core.geometry import grow_tree_3d
    from perfusion.core.network import _junction_geometry

    net = grow_tree_3d((0, 0, 0), (0, 0, 1), root_radius=2.0, min_radius=0.9,
                       length_to_radius=5, ratio=0.8, max_generations=4)
    jl = _junction_geometry(net)
    xyz = net.positions()
    d = xyz[net.seg_end] - xyz[net.seg_start]
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    child = np.flatnonzero(jl["has_parent"])
    turn = np.arccos(np.clip((d[child] * d[jl["parent"][child]]).sum(1), -1, 1))
    assert np.allclose(jl["cos_phi"][child], np.cos(0.75 * turn))

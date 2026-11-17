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

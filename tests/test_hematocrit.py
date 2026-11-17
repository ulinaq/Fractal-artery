"""RBC phase separation (Pries plasma skimming) and hematocrit propagation."""

import numpy as np
import pytest

from perfusion.core.geometry import Network, NodeRole
from perfusion.core.hematocrit import phase_separation_fraction, segment_hematocrit


def test_phase_separation_limits_and_symmetry():
    # symmetric daughters, equal split -> equal RBC split
    assert phase_separation_fraction(0.5, 0.45, 40, 20, 20) == pytest.approx(0.5, abs=1e-9)
    # a low-flow daughter (below X0 = 0.4/D) is skimmed of all cells
    assert phase_separation_fraction(0.005, 0.45, 40, 20, 20) == 0.0
    assert phase_separation_fraction(0.995, 0.45, 40, 20, 20) == 1.0
    # the high-flow branch is enriched, the low-flow branch skimmed (nonlinear split)
    assert phase_separation_fraction(0.7, 0.45, 40, 20, 20) > 0.7
    assert phase_separation_fraction(0.3, 0.45, 40, 20, 20) < 0.3
    # diameter asymmetry shifts the split away from a plain 50/50
    assert phase_separation_fraction(0.5, 0.45, 40, 30, 15) != pytest.approx(0.5, abs=1e-3)


def _y_tree():
    """Trunk (node0->1) that bifurcates at node1 into node2 and node3."""
    return Network(
        node_x=np.array([0.0, 0.0, -1.0, 1.0]),
        node_y=np.array([0.0, 1.0, 2.0, 2.0]),
        node_role=np.array([NodeRole.ROOT, NodeRole.BIFURCATION,
                            NodeRole.TERMINAL, NodeRole.TERMINAL]),
        seg_start=np.array([0, 1, 1]),
        seg_end=np.array([1, 2, 3]),
        seg_length=np.array([1.0, 1.4, 1.4]),
        seg_radius=np.array([2.0, 1.4, 1.0]),
        seg_angle=np.zeros(3),
        root_node=0,
    )


def test_segment_hematocrit_skims_and_conserves():
    net = _y_tree()
    flow = np.array([1.0, 0.9, 0.1])          # daughter seg2 is the low-flow branch
    hd = segment_hematocrit(net, flow, inlet_hd=0.45, radius_to_um=10.0)
    assert hd[0] == pytest.approx(0.45)        # trunk keeps the systemic hematocrit
    assert hd[2] < 0.45 < hd[1]                # skimming: low-flow branch loses cells
    # RBC flux is conserved across the bifurcation
    assert hd[1] * flow[1] + hd[2] * flow[2] == pytest.approx(0.45 * flow[0], rel=1e-6)


def test_run_hybrid_phase_separation_conserves_and_alters_flow():
    from perfusion import Grid, Parameters, run_hybrid

    grid = Grid(nx=40, ny=46, lx=140, ly=160)
    prm = Parameters(r_min_frac=0.4, fahraeus_lindqvist=True)
    dom = (-70, 70, 0, 160)
    base = run_hybrid(prm, grid, dom)                                  # uniform hematocrit
    ps = run_hybrid(prm, grid, dom, phase_separation=True)             # Pries skimming

    n = ps.art.terminal_nodes.size
    inflow = ps.solution.terminal_flux[:n].sum()
    outflow = -ps.solution.terminal_flux[n:].sum()
    assert inflow == pytest.approx(outflow, rel=1e-6)                  # mass still balances
    assert inflow > 0
    # the skimming <-> viscosity coupling perturbs the flux field
    assert not np.allclose(base.flux, ps.flux)


def test_vessel_wall_elasticity_swells_radii_and_conserves():
    from perfusion import Grid, Parameters, run_hybrid

    grid = Grid(nx=40, ny=46, lx=140, ly=160)
    dom = (-70, 70, 0, 160)
    prm = Parameters(r_min_frac=0.4, p_in=10.0, p_out=0.0, compliance=0.02)
    rigid = run_hybrid(prm, grid, dom)                        # compliance ignored (elasticity off)
    elastic = run_hybrid(prm, grid, dom, elasticity=True)     # radii swell with pressure

    n = elastic.art.terminal_nodes.size
    inflow = elastic.solution.terminal_flux[:n].sum()
    outflow = -elastic.solution.terminal_flux[n:].sum()
    assert inflow == pytest.approx(outflow, rel=1e-6)        # mass balances
    assert inflow > 0
    # positive pressure + positive compliance -> arterial radii swell above the rigid baseline
    assert elastic.art.seg_radius.mean() > rigid.art.seg_radius.mean()


def test_single_vessel_hematocrit_unchanged():
    net = Network(
        node_x=np.array([0.0, 0.0]), node_y=np.array([0.0, 1.0]),
        node_role=np.array([NodeRole.ROOT, NodeRole.TERMINAL]),
        seg_start=np.array([0]), seg_end=np.array([1]),
        seg_length=np.array([1.0]), seg_radius=np.array([2.0]), seg_angle=np.zeros(1),
        root_node=0,
    )
    assert segment_hematocrit(net, np.array([1.0]), inlet_hd=0.4)[0] == pytest.approx(0.4)

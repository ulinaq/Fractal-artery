"""Tests for the 3-D hybrid pipeline (3-D trees coupled to a 3-D Darcy bed)."""

import numpy as np
import pytest

from perfusion import Parameters
from perfusion.core.darcy3d import Grid3D
from perfusion.core.geometry import grow_tree_3d
from perfusion.core.pipeline3d import run_hybrid_3d


def _system(max_gen=5):
    art = grow_tree_3d((25, 25, 3), (0, 0, 1), root_radius=2.0, min_radius=0.9,
                       length_to_radius=5, ratio=0.8, max_generations=max_gen)
    ven = grow_tree_3d((25, 25, 47), (0, 0, -1), root_radius=2.0, min_radius=0.9,
                       length_to_radius=5, ratio=0.8, max_generations=max_gen, mirror=True)
    xs = np.concatenate([art.node_x, ven.node_x])
    ys = np.concatenate([art.node_y, ven.node_y])
    zs = np.concatenate([art.node_z, ven.node_z])
    m = 4.0
    dom = (xs.min() - m, xs.max() + m, ys.min() - m, ys.max() + m, zs.min() - m, zs.max() + m)
    grid = Grid3D(nx=22, ny=22, nz=24,
                  lx=dom[1] - dom[0], ly=dom[3] - dom[2], lz=dom[5] - dom[4])
    return art, ven, grid, dom


def test_3d_hybrid_conserves_mass():
    art, ven, grid, dom = _system()
    res = run_hybrid_3d(Parameters(p_in=10.0, p_out=0.0, mu=1.0), grid, art, ven, dom, eps=2.0)
    sol = res.solution
    n_art = res.art.terminal_nodes.size
    inflow = float(sol.terminal_flux[:n_art].sum())
    outflow = float(-sol.terminal_flux[n_art:].sum())
    assert inflow > 0
    assert np.isclose(inflow, outflow, rtol=1e-6)                  # inflow == outflow
    assert np.isclose(sol.terminal_flux.sum(), 0.0, atol=1e-8)     # global mass balance
    assert res.pressure.shape == (grid.nx, grid.ny, grid.nz)


def test_3d_inlets_pinned_and_field_finite():
    art, ven, grid, dom = _system()
    res = run_hybrid_3d(Parameters(p_in=12.0, p_out=2.0, mu=1.0), grid, art, ven, dom, eps=2.0)
    assert np.allclose(res.solution.p_a[art.inlet_nodes], 12.0)     # inlet at p_in
    assert np.allclose(res.solution.p_v[ven.inlet_nodes], 2.0)      # outlet at p_out
    assert np.all(np.isfinite(res.pressure))


def test_multiple_3d_inlets():
    art, ven, grid, dom = _system()
    tips = art.terminal_nodes
    inlets = [art.root_node, int(tips[0])]         # two arterial inlets
    res = run_hybrid_3d(Parameters(p_in=10.0, p_out=0.0, mu=1.0), grid, art, ven, dom,
                        eps=2.0, art_inlets=inlets)
    assert np.allclose(res.solution.p_a[res.art.inlet_nodes], 10.0)
    assert np.isclose(res.solution.terminal_flux.sum(), 0.0, atol=1e-8)


def test_3d_terminals_outside_the_mask_snap_to_tissue():
    """Trees poking out of an organ-shaped 3-D mask still couple (and conserve mass)."""
    from perfusion.core.darcy3d import Grid3D
    from perfusion.core.geometry import grow_tree_3d
    from perfusion.core.parameters import Parameters
    from perfusion.core.pipeline3d import run_hybrid_3d
    from perfusion.treegen import DOMAIN_3D, shape_mask_3d

    mask = shape_mask_3d("ellipsoid", 14, 14, 14, size=0.6)   # small: tips land outside
    art = grow_tree_3d((30, 30, 3), (0, 0, 1), root_radius=2, min_radius=0,
                       length_to_radius=8, max_generations=3)
    ven = grow_tree_3d((30, 30, 57), (0, 0, -1), root_radius=2, min_radius=0,
                       length_to_radius=8, max_generations=3, mirror=True)
    grid = Grid3D(nx=14, ny=14, nz=14, lx=60, ly=60, lz=60, mask=mask)
    res = run_hybrid_3d(Parameters(p_in=10, p_out=0), grid, art, ven, DOMAIN_3D)
    q = res.solution.terminal_flux
    assert np.all(np.isfinite(res.pressure[mask]))
    assert q[q > 0].sum() == pytest.approx(-q[q < 0].sum(), rel=1e-6)
    assert q[q > 0].sum() > 0


def _masked_3d_result():
    from perfusion.treegen import DOMAIN_3D, shape_mask_3d
    mask = shape_mask_3d("ellipsoid", 14, 14, 14, size=0.8)
    art = grow_tree_3d((30, 30, 6), (0, 0, 1), root_radius=2, min_radius=0,
                       length_to_radius=5, max_generations=3)
    ven = grow_tree_3d((30, 30, 54), (0, 0, -1), root_radius=2, min_radius=0,
                       length_to_radius=5, max_generations=3, mirror=True)
    grid = Grid3D(nx=14, ny=14, nz=14, lx=60, ly=60, lz=60, mask=mask)
    return run_hybrid_3d(Parameters(p_in=10, p_out=0), grid, art, ven, DOMAIN_3D), mask


def test_3d_result_has_flux_and_time_of_flight_like_2d():
    res, mask = _masked_3d_result()
    assert res.flux.shape == res.tof.shape == mask.shape
    assert np.all(np.isfinite(res.flux[mask]))
    assert np.all(res.flux[mask] >= 0)
    assert np.isnan(res.flux[~mask]).all()
    tof = res.tof[mask]
    assert np.all(np.isfinite(tof))
    assert np.all(tof > 0)
    assert np.isnan(res.tof[~mask]).all()


def test_3d_face_fluxes_do_not_cross_the_mask():
    from perfusion.core.darcy3d import face_fluxes_3d
    res, mask = _masked_3d_result()
    fx, _, fz = face_fluxes_3d(res.grid, res.pressure)
    assert np.all(fx[~(mask[:-1] & mask[1:])] == 0)
    assert np.all(fz[~(mask[:, :, :-1] & mask[:, :, 1:])] == 0)
    # interior conservation: net face outflow of the active cells equals the net
    # terminal source (+art in, -ven out) — zero overall for a balanced solve
    assert res.solution.terminal_flux.sum() == pytest.approx(0, abs=1e-8)


def test_3d_time_of_flight_is_shortest_at_the_arterial_feeds():
    """Blood reaches the arterial injection cells first, the venous drains later,
    and stagnant tissue last: TOF(art terminals) < TOF(ven terminals) < bulk."""
    from perfusion.core.pipeline3d import _terminal_cells_3d
    res, _ = _masked_3d_result()
    cells = _terminal_cells_3d(res.grid, res.art, res.ven, res.domain_limit)
    na = res.art.terminal_nodes.size
    tof = res.tof.ravel(order="F")
    t_art, t_ven = np.nanmedian(tof[cells[:na]]), np.nanmedian(tof[cells[na:]])
    assert t_art < t_ven < np.nanmedian(tof)


def test_3d_monolithic_coupling_matches_schur():
    res_s, mask = _masked_3d_result()
    from perfusion.core.darcy3d import Grid3D
    grid = Grid3D(nx=14, ny=14, nz=14, lx=60, ly=60, lz=60, mask=mask)
    res_m = run_hybrid_3d(Parameters(p_in=10, p_out=0), grid, res_s.art, res_s.ven,
                          res_s.domain_limit, coupling="monolithic")
    assert np.allclose(res_s.pressure[mask], res_m.pressure[mask], rtol=1e-9, atol=1e-14)
    assert np.allclose(res_s.tof[mask], res_m.tof[mask], rtol=1e-8)
    assert np.allclose(res_s.solution.terminal_flux, res_m.solution.terminal_flux,
                       rtol=1e-9, atol=1e-14)


# --- 3-D analyses: tracer transport, oxygen, kinetic maps, radius adaptation ---------

def _solved_3d():
    art, ven, grid, dom = _system(max_gen=4)
    res = run_hybrid_3d(Parameters(p_in=10.0, p_out=0.0, mu=1.0), grid, art, ven, dom, eps=2.0)
    return res


def test_3d_tracer_transport_conserves_and_washes_out():
    from perfusion.core import tkm, transport

    res = _solved_3d()
    q = res.solution.terminal_flux
    finite = res.tof[np.isfinite(res.tof)]
    dur = 6.0 * float(np.percentile(finite, 80))
    t = np.linspace(0.0, dur, 300)
    aif = tkm.gamma_variate_aif(t, t0=0.05 * dur, A=3.0, B=0.05 * dur)
    frames = transport.transport_forward(res.grid, res.fx, res.fy, res.sources, q, aif,
                                         t[1] - t[0], theta=1.0, fz=res.fz)
    assert frames.shape == (len(t), res.grid.nx, res.grid.ny, res.grid.nz)
    assert np.all(frames >= 0.0)
    mass = frames.reshape(len(t), -1).sum(axis=1)
    assert mass.max() > 0
    assert mass[-1] < 0.5 * mass.max()              # the bolus washes out through the veins
    # explicit-Euler CFL bound is finite and positive in 3-D too
    assert 0 < transport.cfl_limit(res.grid, res.fx, res.fy, res.sources, q, fz=res.fz) < np.inf

    # model-free kinetic maps work voxel-wise in 3-D
    maps = tkm.kinetic_maps(frames, t, aif, mask=res.grid.mask)
    dec = tkm.deconvolution_maps(frames[::6], t[::6], aif[::6], mask=res.grid.mask)
    for m in (maps["Fp"], dec["CBV"]):
        assert m.shape == res.grid.mask.shape
        assert np.isfinite(m[res.grid.mask]).all()


def test_3d_oxygen_extraction():
    from perfusion.core.oxygen import oxygen_from_result

    res = _solved_3d()
    ox = oxygen_from_result(res, k=1.0)
    sat = ox.saturation[res.grid.mask]
    assert ox.concentration.shape == res.grid.mask.shape
    assert np.all((sat >= 0) & (sat <= 1 + 1e-9))
    assert 0.0 < ox.oef < 1.0
    assert oxygen_from_result(res, k=3.0).oef > ox.oef     # faster uptake extracts more
    # diffusion uses the 3-D TPFA Laplacian and stays well-posed
    assert 0.0 < oxygen_from_result(res, k=1.0, diffusivity=0.5).oef < 1.0


def test_3d_adaptation_moves_toward_murray():
    from perfusion.core.pipeline import run_adaptation

    art, ven, grid, dom = _system(max_gen=4)
    ad = run_adaptation(Parameters(p_in=10.0, p_out=0.0, mu=1.0), grid, target_exponent=3.0,
                        domain_limit=dom, art=art, ven=ven, n_iter=25)
    assert ad.art.is_3d
    assert ad.ven.is_3d
    assert abs(ad.murray_exponent - 3.0) <= abs(ad.murray_start - 3.0)
    assert not np.allclose(ad.art.seg_radius, art.seg_radius)      # radii remodelled


def test_3d_reused_coupling_matches_fresh_solve():
    from perfusion.core.pipeline3d import darcy_coupling_3d

    art, ven, grid, dom = _system(max_gen=4)
    prm = Parameters(p_in=10.0, p_out=0.0, mu=1.0)
    for coupling in ("schur", "monolithic"):
        d = darcy_coupling_3d(grid, art, ven, dom, 2.0, coupling)
        a = run_hybrid_3d(prm, grid, art, ven, dom, eps=2.0, darcy=d)
        b = run_hybrid_3d(prm, grid, art, ven, dom, eps=2.0, coupling=coupling)
        assert np.allclose(a.pressure, b.pressure, equal_nan=True)


# ── multi-compartment bed, UQ and calibration on 3-D ──────────────────────────
def _masked_system():
    res, _ = _masked_3d_result()
    return res.art, res.ven, res.grid, res.domain_limit

def test_3d_single_compartment_reduces_to_the_hybrid():
    from perfusion.core.pipeline import run_multicompartment

    art, ven, grid, dom = _masked_system()
    prm = Parameters(p_in=10.0, p_out=0.0, mu=1.0)
    ref = run_hybrid_3d(prm, grid, art, ven, dom, eps=2.0, coupling="schur")
    one = run_multicompartment(prm, grid, permeabilities=(1.0,), betas=(), domain_limit=dom,
                               art=art, ven=ven)
    assert one.compartments.shape == (1, grid.nx, grid.ny, grid.nz)
    np.testing.assert_allclose(one.solution.terminal_flux, ref.solution.terminal_flux,
                               atol=1e-12)
    np.testing.assert_allclose(one.compartments[0], ref.pressure, atol=1e-10)


def test_3d_multicompartment_percolates_and_conserves():
    from perfusion.core.pipeline import run_multicompartment

    art, ven, grid, dom = _masked_system()
    m = run_multicompartment(Parameters(p_in=10.0, p_out=0.0, mu=1.0), grid,
                             permeabilities=(1.0, 0.7, 0.4), betas=(1.0, 1.0),
                             domain_limit=dom, art=art, ven=ven)
    q = m.solution.terminal_flux
    inflow = q[q > 0].sum()
    assert inflow > 0
    assert -q[q < 0].sum() == pytest.approx(inflow, rel=1e-8)
    assert m.perfusion.shape == (2, grid.nx, grid.ny, grid.nz)
    for c in range(2):                       # all the flow crosses every interface
        assert np.nansum(m.perfusion[c]) == pytest.approx(inflow, rel=1e-6)
    assert np.isnan(m.perfusion[0][~grid.mask]).all()
    means = [np.nanmean(np.where(grid.mask, p, np.nan)) for p in m.compartments]
    assert means[0] > means[1] > means[2]    # arteriole → venule pressure cascade


def test_3d_uq_reuses_the_coupling_and_matches_fresh_solves():
    from perfusion import uq

    art, ven, grid, dom = _masked_system()
    r = uq.monte_carlo({"p_in": (5.0, 15.0), "mu": (0.8, 1.5)}, grid, n_samples=6,
                       base={"p_out": 0.0}, domain_limit=dom, art=art, ven=ven)
    assert r.outputs.shape == (6, 3)
    assert r.sensitivity["arterial_inflow"]["p_in"] > 0.5
    assert r.sensitivity["arterial_inflow"]["mu"] < 0
    p_in, mu = r.samples[0]
    fresh = run_hybrid_3d(Parameters(p_in=p_in, p_out=0.0, mu=mu), grid, art, ven, dom)
    tf = fresh.solution.terminal_flux
    assert r.outputs[0, 0] == pytest.approx(tf[tf > 0].sum(), rel=1e-10)
    s = uq.sobol_indices({"p_in": (5.0, 15.0), "permeability": (0.5, 2.0)}, grid,
                         n_samples=10, base={"p_out": 0.0}, domain_limit=dom, art=art, ven=ven)
    assert s.fit_r2["arterial_inflow"] > 0.9
    with pytest.raises(ValueError, match="cannot vary on a 3-D model"):
        uq.monte_carlo({"r_min_frac": (0.3, 0.6)}, grid, n_samples=4, domain_limit=dom,
                       art=art, ven=ven)


@pytest.mark.parametrize(("param", "truth", "bounds"),
                         [("p_in", 13.0, (1.0, 40.0)), ("permeability", 2.5, (0.2, 8.0))])
def test_3d_calibration_recovers_the_input(param, truth, bounds):
    from dataclasses import replace

    from perfusion import inverse

    art, ven, grid, dom = _masked_system()
    kw = {"p_in": 10.0, "p_out": 0.0}
    g = grid
    if param == "permeability":
        g = replace(grid, permeability=truth)
    else:
        kw[param] = truth
    tf = run_hybrid_3d(Parameters(**kw), g, art, ven, dom).solution.terminal_flux
    target = float(tf[tf > 0].sum())
    c = inverse.calibrate(target, param, grid, bounds, base={"p_in": 10.0, "p_out": 0.0},
                          domain_limit=dom, art=art, ven=ven)
    assert c.value == pytest.approx(truth, rel=1e-3)
    assert c.achieved == pytest.approx(target, rel=1e-5)


def test_3d_shear_thinning_conserves_mass_and_thickens_blood():
    art, ven, grid, dom = _system(max_gen=4)
    newt = run_hybrid_3d(Parameters(p_in=10.0, p_out=0.0), grid, art, ven, dom, eps=2.0)
    thin = run_hybrid_3d(Parameters(p_in=10.0, p_out=0.0, shear_thinning=True), grid,
                         art, ven, dom, eps=2.0)
    n_art = thin.art.terminal_nodes.size
    tf = thin.solution.terminal_flux
    assert np.isclose(tf[:n_art].sum(), -tf[n_art:].sum(), rtol=1e-8)
    assert thin.rheology["converged"]
    assert thin.rheology["factor_max"] > 1.0
    assert 0 < tf[:n_art].sum() < newt.solution.terminal_flux[:n_art].sum()
    assert newt.rheology is None


def test_3d_casson_yield_stress_blood():
    art, ven, grid, dom = _system(max_gen=4)
    newt = run_hybrid_3d(Parameters(p_in=10.0, p_out=0.0), grid, art, ven, dom, eps=2.0)
    cas = run_hybrid_3d(Parameters(p_in=10.0, p_out=0.0, shear_thinning=True,
                                   blood_law="casson", casson_yield=0.001),
                        grid, art, ven, dom, eps=2.0)
    n_art = cas.art.terminal_nodes.size
    tf = cas.solution.terminal_flux
    assert cas.rheology["converged"]
    assert np.isclose(tf[:n_art].sum(), -tf[n_art:].sum(), rtol=1e-8)
    assert 0 < tf[:n_art].sum() < newt.solution.terminal_flux[:n_art].sum()


def test_3d_honours_the_rheology_and_flow_options():
    # run_hybrid_3d used to solve linearly with the scalar mu whatever was asked.
    import warnings

    art, ven, grid, dom = _system()
    base = run_hybrid_3d(Parameters(p_in=10.0, p_out=0.0, mu=1.0), grid, art, ven, dom, eps=2.0)
    q0 = float(base.solution.q_a[0])
    variants = {
        "fl": (Parameters(p_in=10.0, p_out=0.0, mu=1.0, fahraeus_lindqvist=True,
                          radius_to_um=10.0), {}),
        "elastic": (Parameters(p_in=10.0, p_out=0.0, mu=1.0, compliance=0.05),
                    {"elasticity": True}),
    }
    for name, (prm, kw) in variants.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = run_hybrid_3d(prm, grid, art, ven, dom, eps=2.0, **kw)
        sol = res.solution
        assert not np.isclose(float(sol.q_a[0]), q0, rtol=1e-6), name
        assert np.isclose(sol.terminal_flux.sum(), 0.0, atol=1e-8), name

    # the nonlinear option really runs the junction-loss fixed point
    from perfusion.core import network
    from perfusion.core.pipeline3d import darcy_coupling_3d

    prm = Parameters(p_in=10.0, p_out=0.0, mu=1.0, rho=1e4)
    coupling = darcy_coupling_3d(grid, art, ven, dom, eps=2.0)
    res = run_hybrid_3d(prm, grid, art, ven, dom, eps=2.0, darcy=coupling, nonlinear=True)
    ref = network.solve_nonlinear(art, ven, coupling[1], 10.0, 0.0, prm, 1.0)
    assert np.allclose(res.solution.q_a, ref.q_a, rtol=1e-10)
    assert not np.array_equal(res.solution.q_a, base.solution.q_a)

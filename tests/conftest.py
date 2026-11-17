"""Shared pytest fixtures for the perfusion test suite."""

from pathlib import Path

import numpy as np
import pytest

from perfusion import Grid, Parameters, arterial_tree, run_hybrid, venous_tree

TEST_DATA = Path(__file__).resolve().parent / "data"


@pytest.fixture
def params() -> Parameters:
    """Default parameters with a fixed seed (deterministic deep tree)."""
    return Parameters(seed=0)


@pytest.fixture
def small_params() -> Parameters:
    """Shallow tree for fast solver/topology tests."""
    return Parameters(r_min_frac=0.5, seed=0)


@pytest.fixture
def grid() -> Grid:
    """Capillary-bed grid matching the default domain limits."""
    return Grid(nx=140, ny=160, lx=140, ly=160)


@pytest.fixture
def small_grid() -> Grid:
    """Small grid for cheap Darcy/coupling tests."""
    return Grid(nx=40, ny=40, lx=1.0, ly=1.0)


@pytest.fixture
def masked_grid() -> Grid:
    """Grid with an inactive border (exercises the domain mask)."""
    mask = np.zeros((30, 30), dtype=bool)
    mask[8:22, 8:22] = True
    return Grid(nx=30, ny=30, mask=mask)


@pytest.fixture
def art(small_params) -> "object":
    return arterial_tree(small_params)


@pytest.fixture
def ven(small_params) -> "object":
    return venous_tree(small_params)


@pytest.fixture(scope="session")
def hybrid_result():
    """A full hybrid solve, computed once and shared across tests."""
    return run_hybrid(Parameters(seed=0), Grid(nx=140, ny=160, lx=140, ly=160))


@pytest.fixture(scope="session")
def reference_hybrid():
    """Locked reference summaries from ``scripts/generate_data.py``."""
    with np.load(TEST_DATA / "reference_hybrid.npz") as data:
        return {k: data[k] for k in data.files}


@pytest.fixture
def flow_field(small_grid):
    """A simple Darcy flow with one arterial source and one venous sink.

    Returns ``(grid, fx, fy, sources, terminal_flux)`` for transport / TOF tests.
    """
    from perfusion import darcy

    g = small_grid
    terminals = g.sub2ind([10, 30], [10, 30])          # inject at 10,10; extract at 30,30
    _, field_response, sources = darcy.terminal_response(g, terminals, eps=3.0)
    terminal_flux = np.array([1.0, -1.0])
    pressure = darcy.reconstruct_field(field_response, terminal_flux)
    pressure = pressure.reshape(g.nx, g.ny, order="F")
    _, tx, ty = darcy.tpfa_operator(g)
    fx, fy = darcy.face_fluxes(g, pressure, tx, ty)
    return g, fx, fy, sources, terminal_flux

"""Smoke tests for the plotting helpers (Agg backend, no display)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from perfusion import viz


def test_plot_network(art):
    fig, ax = plt.subplots()
    viz.plot_network(ax, art, color="red")
    # segments are drawn as a single LineCollection
    coll = ax.collections[-1]
    assert len(coll.get_segments()) == art.n_segments
    plt.close(fig)


def test_plot_network_colored_by_flow(art):
    import numpy as np

    fig, ax = plt.subplots()
    values = np.abs(np.arange(art.n_segments, dtype=float))
    lc = viz.plot_network(ax, art, values=values, label="|q|")
    assert lc.get_array() is not None
    plt.close(fig)


def test_plot_tkm_and_bars():
    import numpy as np

    from perfusion import tkm

    t = np.arange(0, 60, 0.5)
    aif = tkm.gamma_variate_aif(t, t0=3.0)
    tissue = tkm.one_compartment(t, aif, 40.0, 8.0)
    fits = {"1CM": tissue, "MS": None}
    fig, (a0, a1) = plt.subplots(1, 2)
    viz.plot_tkm(a0, t, aif, tissue, fits)
    viz.plot_parameter_bars(a1, {"1CM": 40.0, "MS": 8.0}, truth=40.0)
    assert a0.get_legend() is not None
    plt.close(fig)


def test_plot_field(hybrid_result):
    fig, ax = plt.subplots()
    im = viz.plot_field(ax, hybrid_result.pressure, title="P")
    assert im is not None
    assert ax.get_title() == "P"
    plt.close(fig)

"""Transient (pulsatile) Windkessel reduction: compliance damps the pulse."""

import numpy as np
import pytest

from perfusion.core import transient


def _pulsatility(a):
    a = np.asarray(a, float)
    return (a.max() - a.min()) / a.mean()


def test_compliance_zero_is_quasi_static():
    # with no compliance the tissue flow follows the inlet pulse (no damping)
    _, p_in = transient.pulsatile_inlet(10.0, 4.0, 1.0, 0.01, n_cycles=3)
    wk = transient.windkessel(p_in, r_total=5.0, compliance_total=0.0, dt=0.01)
    last = slice(-100, None)
    # q_tissue = (P - p_out)/Rd is an affine map of p_in here -> same *relative* pulse
    assert _pulsatility(wk["q_tissue"][last]) == pytest.approx(_pulsatility(p_in[last]), rel=1e-6)


def test_compliance_damps_the_pulse():
    _, p_in = transient.pulsatile_inlet(10.0, 4.0, 1.0, 0.01, n_cycles=6)
    last = slice(-100, None)
    stiff = transient.windkessel(p_in, 5.0, 0.05, 0.01)
    soft = transient.windkessel(p_in, 5.0, 2.0, 0.01)     # much more compliant
    # more compliance -> smaller tissue-flow pulsatility (Windkessel buffering)
    assert _pulsatility(soft["q_tissue"][last]) < _pulsatility(stiff["q_tissue"][last])
    assert _pulsatility(soft["q_tissue"][last]) < _pulsatility(p_in[last])


def test_reaches_periodic_steady_state():
    _, p_in = transient.pulsatile_inlet(10.0, 4.0, 1.0, 0.01, n_cycles=8)
    q = transient.windkessel(p_in, 5.0, 1.0, 0.01)["q_tissue"]
    npc = 100
    assert np.allclose(q[-npc:], q[-2 * npc:-npc], atol=1e-3)   # last two cycles coincide


def test_compliance_total_scales_with_geometry():
    from perfusion import arterial_tree
    from perfusion.core.parameters import Parameters

    art = arterial_tree(Parameters(r_min_frac=0.5))
    c1 = transient.compliance_total([art], 0.01)
    c2 = transient.compliance_total([art], 0.02)
    assert c1 > 0
    assert c2 == pytest.approx(2 * c1, rel=1e-9)                # linear in compliance

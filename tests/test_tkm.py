"""Tests for tracer-kinetic models and fitting (Paper C)."""

import numpy as np
import pytest

from perfusion import tkm


@pytest.fixture
def timeline():
    t = np.arange(0.0, 120.0, 0.5)
    aif = tkm.gamma_variate_aif(t, t0=5.0, A=3.0, B=1.5)
    return t, aif


def test_gamma_variate_shape(timeline):
    t, aif = timeline
    assert aif.shape == t.shape
    assert np.all(aif >= 0)
    assert aif[t < 5.0].max() == 0.0            # zero before bolus arrival
    assert aif.argmax() > 0


def test_one_compartment_positive_and_causal(timeline):
    t, aif = timeline
    C = tkm.one_compartment(t, aif, Fp=40.0, vp=8.0)
    assert C.shape == t.shape
    assert np.all(C >= 0)
    assert C[0] == pytest.approx(0.0, abs=1e-9)


def test_fit_1cm_recovers_truth(timeline):
    t, aif = timeline
    C = tkm.one_compartment(t, aif, Fp=40.0, vp=8.0)
    r = tkm.fit(t, aif, C, "1CM")
    assert r.Fp == pytest.approx(40.0, rel=1e-2)
    assert r.vp == pytest.approx(8.0, rel=1e-2)


def test_fit_2cum_and_2cxm_reduce_to_1cm(timeline):
    # With no interstitial leak the two-compartment models recover the 1CM flow.
    t, aif = timeline
    C = tkm.one_compartment(t, aif, Fp=35.0, vp=6.0)
    for model in ("2CUM", "2CXM"):
        r = tkm.fit(t, aif, C, model)
        assert r.Fp == pytest.approx(35.0, rel=5e-2)


def test_maximum_slope_returns_flow(timeline):
    t, aif = timeline
    C = tkm.one_compartment(t, aif, Fp=40.0, vp=8.0)
    r = tkm.fit(t, aif, C, "MS")
    assert r.Fp > 0
    assert np.isnan(r.vp)                        # MS does not estimate volume


def test_fit_unknown_model_raises(timeline):
    t, aif = timeline
    with pytest.raises(ValueError, match="unknown model"):
        tkm.fit(t, aif, aif, "nope")


def test_two_compartment_exchange_curve(timeline):
    t, aif = timeline
    C = tkm.two_compartment_exchange(t, aif, Fp=30.0, vp=5.0, PS=2.0, ve=10.0)
    assert C.shape == t.shape
    assert np.all(np.isfinite(C))


def test_kinetic_maps_perfusion_and_mask():
    import numpy as np

    from perfusion.core.tkm import gamma_variate_aif, kinetic_maps, one_compartment

    t = np.linspace(0, 10, 120)
    ca = gamma_variate_aif(t, t0=0.5, A=3.0, B=1.0)
    nx, ny = 3, 2
    frames = np.zeros((len(t), nx, ny))
    frames[:, 0, 0] = one_compartment(t, ca, Fp=2.0, vp=0.2)   # well perfused
    frames[:, 1, 0] = one_compartment(t, ca, Fp=0.5, vp=0.2)   # poorly perfused
    mask = np.ones((nx, ny), bool)
    mask[2, 1] = False

    m = kinetic_maps(frames, t, ca, mask=mask)
    assert m["Fp"].shape == (nx, ny)
    assert m["Fp"][0, 0] > m["Fp"][1, 0]        # higher flow -> higher max-slope Fp
    assert np.isnan(m["Fp"][2, 1])              # inactive voxel masked out


def test_deconvolution_recovers_flow_and_transit_time():
    import numpy as np

    from perfusion.core.tkm import (
        deconvolution_maps,
        deconvolve,
        gamma_variate_aif,
        one_compartment,
    )

    # well-resolved system (~40 samples per MTT) so the SVD is accurate
    t = np.linspace(0, 40, 800)
    ca = gamma_variate_aif(t, t0=2.0, A=3.0, B=2.0)
    ca = ca / ca.max()
    Fp_true, mtt_true = 0.5, 2.0
    curve = one_compartment(t, ca, Fp=Fp_true, vp=Fp_true * mtt_true)

    k = deconvolve(t, ca, curve, oi=0.01).ravel()
    fp = k.max()
    cbv = np.trapezoid(k, t)
    mtt = cbv / fp
    assert abs(fp - Fp_true) / Fp_true < 0.15         # residue peak ~ Fp
    assert abs(mtt - mtt_true) / mtt_true < 0.2       # central-volume theorem
    assert abs(cbv - Fp_true * mtt_true) / (Fp_true * mtt_true) < 0.1

    # map form: higher-flow voxel maps to higher Fp; masked voxel is NaN
    nx, ny = 3, 2
    frames = np.zeros((len(t), nx, ny))
    frames[:, 0, 0] = one_compartment(t, ca, Fp=0.8, vp=0.8 * mtt_true)
    frames[:, 1, 0] = one_compartment(t, ca, Fp=0.3, vp=0.3 * mtt_true)
    mask = np.ones((nx, ny), bool)
    mask[2, 1] = False
    m = deconvolution_maps(frames, t, ca, mask=mask, oi=0.01)
    assert set(m) == {"Fp", "CBV", "MTT"}
    assert m["Fp"][0, 0] > m["Fp"][1, 0]
    assert np.isnan(m["Fp"][2, 1])
    assert np.isnan(m["MTT"][2, 1])


def test_osvd_is_delay_insensitive():
    import numpy as np

    from perfusion.core.tkm import deconvolve, gamma_variate_aif, one_compartment

    t = np.linspace(0, 40, 400)
    ca = gamma_variate_aif(t, t0=3.0, A=3.0, B=1.5)
    ca = ca / ca.max()
    curve = one_compartment(t, ca, Fp=0.5, vp=0.5 * 3.0)
    lag = 8
    delayed = np.concatenate([np.zeros(lag), curve[:-lag]])   # tissue bolus arrives late

    def peak(c, circ):
        return deconvolve(t, ca, c, oi=0.02, circulant=circ).ravel().max()

    ssvd_shift = abs(peak(delayed, False) - peak(curve, False))
    osvd_shift = abs(peak(delayed, True) - peak(curve, True))
    # block-circulant oSVD is (near-)invariant to a bolus arrival delay; sSVD is not
    assert osvd_shift < 0.02
    assert osvd_shift < ssvd_shift

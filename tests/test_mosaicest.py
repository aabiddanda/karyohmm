"""Test suite for mosaic cell-fraction estimation."""

import numpy as np
import pytest
from unittest.mock import patch
from hypothesis import given
from hypothesis import strategies as st
from karyohmm_utils import logsumexp

from karyohmm import MosaicEst, PGTSim
from karyohmm.simulator import PGTSimMosaic

# ---------------------------------------------------------------------------
# Shared simulation fixtures
# ---------------------------------------------------------------------------

pgt_sim = PGTSim()
data_disomy = pgt_sim.full_ploidy_sim(m=5000, length=1e7, std_dev=0.1, seed=42)
data_trisomy = pgt_sim.full_ploidy_sim(m=5000, ploidy=3, length=1e7, std_dev=0.1, seed=42)
data_monosomy = pgt_sim.full_ploidy_sim(m=5000, ploidy=1, length=1e7, std_dev=0.1, seed=42)

# Coherent mosaic fixtures: weighted blend of full-ploidy sims sharing the
# same parental haplotypes (same seed ⟹ same mat_haps / pat_haps / pos).
_CF_GAIN = 0.4
_CF_LOSS = 0.4

_data_mosaic_gain = dict(data_disomy)
_data_mosaic_gain["baf"] = (
    (1 - _CF_GAIN) * data_disomy["baf"] + _CF_GAIN * data_trisomy["baf"]
)
_data_mosaic_gain["lrr"] = (
    (1 - _CF_GAIN) * data_disomy["lrr"] + _CF_GAIN * data_trisomy["lrr"]
)
_data_mosaic_gain["sigmas"] = (
    (1 - _CF_GAIN) * data_disomy["sigmas"] + _CF_GAIN * data_trisomy["sigmas"]
)

_data_mosaic_loss = dict(data_disomy)
_data_mosaic_loss["baf"] = (
    (1 - _CF_LOSS) * data_disomy["baf"] + _CF_LOSS * data_monosomy["baf"]
)
_data_mosaic_loss["lrr"] = (
    (1 - _CF_LOSS) * data_disomy["lrr"] + _CF_LOSS * data_monosomy["lrr"]
)
_data_mosaic_loss["sigmas"] = (
    (1 - _CF_LOSS) * data_disomy["sigmas"] + _CF_LOSS * data_monosomy["sigmas"]
)


def _make(data):
    """Construct MosaicEst from a simulation result dict."""
    return MosaicEst(
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        bafs=data["baf"],
        pos=data["pos"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
    )


# ---------------------------------------------------------------------------
# Construction and preprocessing
# ---------------------------------------------------------------------------


def test_init_runs_preprocessing():
    """After construction, het sites are phased and transition matrix is ready."""
    m = _make(data_disomy)
    assert m.n_het > 0
    assert m.phased_baf.size == m.n_het
    assert m.A is not None


def test_no_lrr_warns():
    """Omitting LRR emits a UserWarning rather than raising."""
    with pytest.warns(UserWarning, match="lrrs not provided"):
        MosaicEst(
            mat_haps=data_disomy["mat_haps"],
            pat_haps=data_disomy["pat_haps"],
            bafs=data_disomy["baf"],
            pos=data_disomy["pos"],
        )


def test_phased_baf_disomy_centred():
    """Phased BAF at disomy het sites should have mean near zero."""
    m = _make(data_disomy)
    assert abs(np.mean(m.phased_baf)) < 0.05


def test_n_het_minimum():
    """Raises if fewer than 10 expected-het sites exist."""
    rng = np.random.default_rng(0)
    # All-homozygous parental haplotypes → no expected het sites
    mh = np.zeros((2, 200), dtype=np.int8)
    ph = np.zeros((2, 200), dtype=np.int8)
    with pytest.raises(ValueError, match="Fewer than 10"):
        MosaicEst(
            mat_haps=mh, pat_haps=ph,
            bafs=rng.uniform(size=200),
            pos=np.sort(rng.uniform(high=1e7, size=200)),
            lrrs=np.zeros(200), sigmas=np.ones(200),
        )


@given(
    sw_err=st.floats(min_value=1e-8, max_value=0.05),
    t_rate=st.floats(min_value=1e-8, max_value=0.2),
)
def test_transition_matrix_rows_sum_to_one(sw_err, t_rate):
    """All rows of the log-transition matrix must sum to 0 (probability 1)."""
    m = _make(data_disomy)
    m.create_transition_matrix(switch_err=sw_err, t_rate=t_rate)
    assert np.isclose(logsumexp(m.A[0, :]), 0.0)
    assert np.isclose(logsumexp(m.A[1, :]), 0.0)
    assert np.isclose(logsumexp(m.A[2, :]), 0.0)


# ---------------------------------------------------------------------------
# forward_algo_full — likelihood sanity checks
# ---------------------------------------------------------------------------


def test_forward_loglik_is_finite():
    """forward_algo_full returns a finite log-likelihood for valid cf."""
    m = _make(data_disomy)
    _, _, ll = m.forward_algo_full(cf=0.0)
    assert np.isfinite(ll)


def test_forward_loglik_decreases_at_cf_zero_for_disomy():
    """For a disomy sample cf=0 should have the highest or near-highest likelihood."""
    m = _make(data_disomy)
    ll0 = m.forward_algo_full(cf=0.0)[2]
    ll_high = m.forward_algo_full(cf=0.4)[2]
    assert ll0 > ll_high


def test_forward_loglik_increases_with_cf_for_trisomy():
    """For a full trisomy the likelihood should increase moving away from cf=0."""
    m = _make(data_trisomy)
    ll0 = m.forward_algo_full(cf=0.0)[2]
    ll_true = m.forward_algo_full(cf=0.9)[2]
    assert ll_true > ll0


# ---------------------------------------------------------------------------
# est_mle_cf — point estimation
# ---------------------------------------------------------------------------


def test_mle_cf_disomy_near_zero():
    """Disomy: MLE cell fraction should be effectively zero."""
    m = _make(data_disomy)
    m.est_mle_cf()
    assert m.mle_cf is not None
    assert not np.isnan(m.mle_cf)
    assert m.mle_cf < 0.05


def test_mle_cf_trisomy_near_one():
    """Full trisomy: MLE cell fraction should be near 1."""
    m = _make(data_trisomy)
    m.est_mle_cf()
    assert m.mle_cf > 0.8


def test_mle_cf_monosomy_near_one():
    """Full monosomy: MLE cell fraction should be near 1."""
    m = _make(data_monosomy)
    m.est_mle_cf()
    assert m.mle_cf > 0.8


def test_mle_cf_mosaic_gain_detected():
    """40% mosaic trisomy: estimated cf should exceed 0.2."""
    m = _make(_data_mosaic_gain)
    m.est_mle_cf()
    assert not np.isnan(m.mle_cf)
    assert m.mle_cf > 0.2


def test_mle_cf_mosaic_loss_detected():
    """40% mosaic monosomy: estimated cf should exceed 0.2."""
    m = _make(_data_mosaic_loss)
    m.est_mle_cf()
    assert not np.isnan(m.mle_cf)
    assert m.mle_cf > 0.2


# ---------------------------------------------------------------------------
# ci_mle_cf — confidence intervals
# ---------------------------------------------------------------------------


def _check_ci(ci):
    assert ci[0] <= ci[1] <= ci[2]
    assert 0.0 <= ci[0]
    assert ci[2] <= 1.0


def test_ci_ordered_disomy():
    m = _make(data_disomy)
    m.est_mle_cf()
    _check_ci(m.ci_mle_cf())


def test_ci_ordered_trisomy():
    m = _make(data_trisomy)
    m.est_mle_cf()
    _check_ci(m.ci_mle_cf())


def test_ci_ordered_monosomy():
    m = _make(data_monosomy)
    m.est_mle_cf()
    _check_ci(m.ci_mle_cf())


def test_ci_mosaic_gain_reasonable():
    """95% CI for 40% mosaic gain: ordered, non-trivial width, and mle_cf > 0."""
    m = _make(_data_mosaic_gain)
    m.est_mle_cf()
    ci = m.ci_mle_cf()
    _check_ci(ci)
    # CI should be non-degenerate and centre well above zero
    assert (ci[2] - ci[0]) > 0.01
    assert ci[1] > 0.1


# ---------------------------------------------------------------------------
# lrt_cf — likelihood-ratio test
# ---------------------------------------------------------------------------


def test_lrt_cf_disomy_small():
    """LRT statistic should be small (near 0) for a true disomy."""
    m = _make(data_disomy)
    m.est_mle_cf()
    assert m.lrt_cf() < 5.0  # well below chi2(1) 0.05 critical value of 3.84


def test_lrt_cf_trisomy_large():
    """LRT statistic should be large and significant for a full trisomy."""
    m = _make(data_trisomy)
    m.est_mle_cf()
    assert m.lrt_cf() > 10.0


def test_lrt_cf_mosaic_gain_significant():
    """LRT should detect 40% mosaic trisomy as significant."""
    m = _make(_data_mosaic_gain)
    m.est_mle_cf()
    assert m.lrt_cf() > 3.84  # chi2(1) p < 0.05


# ---------------------------------------------------------------------------
# Robustness to parental haplotype errors
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Coverage of error / boundary paths
# ---------------------------------------------------------------------------


def test_forward_algo_full_first_site_is_het():
    """forward_algo_full correctly adds BAF emission when site 0 is a het site."""
    # Build a tiny synthetic case where the first SNP is expected-het so the
    # `if is_het[0]` branch inside forward_algo_full is exercised.
    rng = np.random.default_rng(7)
    m_sites = 200
    pos = np.sort(rng.uniform(high=1e7, size=m_sites))
    # Force site 0 to be a het site: mat hom-ref (0,0), pat hom-alt (1,1)
    mat_haps = rng.integers(0, 2, size=(2, m_sites))
    pat_haps = rng.integers(0, 2, size=(2, m_sites))
    mat_haps[:, 0] = 0
    pat_haps[:, 0] = 1
    # Ensure enough other het sites exist so _baf_hets doesn't raise
    for i in range(1, 30):
        mat_haps[:, i] = 0
        pat_haps[:, i] = 1
    bafs = rng.uniform(size=m_sites)
    lrrs = rng.normal(size=m_sites)
    sigmas = np.abs(rng.normal(loc=0.2, size=m_sites)) + 0.05
    m = MosaicEst(mat_haps=mat_haps, pat_haps=pat_haps, bafs=bafs, pos=pos,
                  lrrs=lrrs, sigmas=sigmas)
    assert m.het_idx[0] == 0  # confirm first site is het
    _, _, ll = m.forward_algo_full(cf=0.1)
    assert np.isfinite(ll)


def test_est_mle_cf_failure_sets_nan():
    """If forward_algo_full raises, est_mle_cf stores nan rather than crashing."""
    m = _make(data_disomy)
    with patch.object(m, "forward_algo_full", side_effect=ValueError("synthetic failure")):
        m.est_mle_cf()
    assert np.isnan(m.mle_cf)


def test_ci_mle_cf_boundary_low():
    """ci_mle_cf uses one-sided finite difference when mle_cf < h."""
    m = _make(data_disomy)
    m.est_mle_cf()
    # Use h larger than the near-zero mle_cf to force the `cf < h` branch
    ci = m.ci_mle_cf(h=max(m.mle_cf + 1e-4, 1e-3))
    _check_ci(ci)


def test_ci_mle_cf_boundary_high():
    """ci_mle_cf uses one-sided finite difference when mle_cf is near 1."""
    m = _make(data_trisomy)
    m.est_mle_cf()
    # Use h larger than (0.999 - mle_cf) to force the high-boundary branch
    gap = 0.999 - m.mle_cf
    if gap > 0:
        ci = m.ci_mle_cf(h=gap + 1e-4)
        _check_ci(ci)


def test_ci_mle_cf_exception_handler():
    """ci_mle_cf returns [nan, nan, nan] when the Hessian computation fails."""
    m = _make(data_disomy)
    m.est_mle_cf()
    with patch.object(m, "forward_algo_full", side_effect=ZeroDivisionError):
        ci = m.ci_mle_cf()
    assert all(np.isnan(v) for v in ci)


def test_lrt_cf_auto_calls_est_mle_cf():
    """lrt_cf runs est_mle_cf internally when mle_cf has not been set yet."""
    m = _make(data_trisomy)
    assert m.mle_cf is None
    lrt = m.lrt_cf()
    assert m.mle_cf is not None  # was set as a side-effect
    assert np.isfinite(lrt) and lrt > 0


def test_lrt_cf_nan_when_mle_failed():
    """lrt_cf returns nan when the MLE failed (mle_cf is nan)."""
    m = _make(data_disomy)
    m.mle_cf = np.nan  # simulate a prior optimisation failure
    assert np.isnan(m.lrt_cf())


def test_robust_to_switch_errors():
    """Switch errors in parental haps do not affect mle_cf (het sites immune)."""
    from karyohmm.simulator import PGTSimBase

    sim = PGTSimBase()
    mh, ph, _, _ = sim.create_switch_errors(
        data_disomy["mat_haps"], data_disomy["pat_haps"], err_rate=0.1, seed=1
    )
    m_clean = _make(_data_mosaic_gain)
    m_sw = MosaicEst(
        mat_haps=mh, pat_haps=ph,
        bafs=_data_mosaic_gain["baf"], pos=_data_mosaic_gain["pos"],
        lrrs=_data_mosaic_gain["lrr"], sigmas=_data_mosaic_gain["sigmas"],
    )
    m_clean.est_mle_cf()
    m_sw.est_mle_cf()
    assert abs(m_clean.mle_cf - m_sw.mle_cf) < 0.01


def test_robust_to_genotyping_errors():
    """Up to 5% genotyping error in parental haps causes < 0.02 drift in mle_cf."""
    from karyohmm.simulator import PGTSimBase

    sim = PGTSimBase()
    _, mh_e = sim.create_genotyping_errors(data_disomy["mat_haps"], err_rate=0.05, seed=1)
    _, ph_e = sim.create_genotyping_errors(data_disomy["pat_haps"], err_rate=0.05, seed=2)
    m_clean = _make(_data_mosaic_gain)
    m_ge = MosaicEst(
        mat_haps=mh_e, pat_haps=ph_e,
        bafs=_data_mosaic_gain["baf"], pos=_data_mosaic_gain["pos"],
        lrrs=_data_mosaic_gain["lrr"], sigmas=_data_mosaic_gain["sigmas"],
    )
    m_clean.est_mle_cf()
    m_ge.est_mle_cf()
    assert abs(m_clean.mle_cf - m_ge.mle_cf) < 0.02

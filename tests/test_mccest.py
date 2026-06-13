"""Test suite for karyoHMM DuoHMM."""

from karyohmm import MccEst, PGTSim
import numpy as np

# --- Generating test data for applications in the DuoHMM setting --- #
pgt_sim = PGTSim()
data_disomy = pgt_sim.full_ploidy_sim(m=1000, mix_prop=0.01, std_dev=0.1, seed=42)


def test_init():
    """Test the initialization of the estimator."""
    _ = MccEst()


def test_loglik_trio(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    pat_haps=np.array([[1, 1], [1, 1]]),
):
    mcc = MccEst()
    ll = mcc.loglik_mcc_trio(
        bafs=bafs, mat_haps=mat_haps, pat_haps=pat_haps, c=0.1, std_dev=0.1
    )
    assert ~np.isnan(ll)


def test_loglik_poc(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    freqs=np.array([0.3, 0.3]),
):
    mcc = MccEst()
    ll = mcc.loglik_mcc_poc(
        bafs=bafs, mat_haps=mat_haps, freqs=freqs, c=0.1, std_dev=0.1
    )
    assert ~np.isnan(ll)


def test_mle_trio(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    pat_haps=np.array([[1, 1], [1, 1]]),
):
    mcc = MccEst()
    (c_est, std_dev) = mcc.est_mcc_trio(bafs=bafs, mat_haps=mat_haps, pat_haps=pat_haps)
    assert (c_est >= 0) and (c_est <= 0.5)


def test_mle_poc(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    freqs=np.array([0.3, 0.3]),
):
    mcc = MccEst()
    (c_est, std_dev) = mcc.est_mcc_poc(bafs=bafs, mat_haps=mat_haps, freqs=freqs)
    assert (c_est >= 0) and (c_est <= 0.5)


def test_ci_trio(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    pat_haps=np.array([[1, 1], [1, 1]]),
):
    mcc = MccEst()
    (c_est, std_dev) = mcc.est_mcc_trio(bafs=bafs, mat_haps=mat_haps, pat_haps=pat_haps)
    (lower95_c, x, upper95_c) = mcc.mcc_ci_trio(
        bafs=bafs, mat_haps=mat_haps, pat_haps=pat_haps, c_hat=c_est, std_dev=std_dev
    )
    assert x == c_est
    assert lower95_c <= upper95_c


def test_ci_poc(
    bafs=np.array([0.6, 0.5]),
    mat_haps=np.array([[0, 0], [0, 0]]),
    freqs=np.array([0.3, 0.3]),
):
    mcc = MccEst()
    (c_est, std_dev) = mcc.est_mcc_poc(bafs=bafs, mat_haps=mat_haps, freqs=freqs)
    (lower95_c, x, upper95_c) = mcc.mcc_ci_poc(
        bafs=bafs, mat_haps=mat_haps, freqs=freqs, c_hat=c_est, std_dev=std_dev
    )
    assert x == c_est
    assert lower95_c <= upper95_c


# --- Shared small dataset used by boundary / edge-case tests ---
_pgt_small = PGTSim()
_d_small = _pgt_small.full_ploidy_sim(m=500, mix_prop=0.0, std_dev=0.1, seed=7)
_bafs_s = _d_small["baf"]
_mat_s = _d_small["mat_haps"]
_pat_s = _d_small["pat_haps"]
_freq_s = _d_small["af"]


# ---------------------------------------------------------------------------
# CI boundary / fallback tests
# (cover the ValueError exception handlers in mcc_ci_poc and mcc_ci_trio)
# ---------------------------------------------------------------------------


def test_ci_poc_lower_fallback():
    """mcc_ci_poc with c_hat=0 falls back to lower_CI=0 without raising."""
    mcc = MccEst()
    lower, x, upper = mcc.mcc_ci_poc(_bafs_s, _mat_s, _freq_s, c_hat=0.0, std_dev=0.1)
    assert lower == 0.0
    assert x == 0.0
    assert upper >= 0.0


def test_ci_poc_upper_fallback():
    """mcc_ci_poc with c_hat=0.5 falls back to upper_CI=0.5 without raising."""
    mcc = MccEst()
    lower, x, upper = mcc.mcc_ci_poc(_bafs_s, _mat_s, _freq_s, c_hat=0.5, std_dev=0.1)
    assert upper == 0.5
    assert x == 0.5


def test_ci_trio_lower_fallback():
    """mcc_ci_trio with c_hat=0 falls back to lower_CI=0 without raising."""
    mcc = MccEst()
    lower, x, upper = mcc.mcc_ci_trio(_bafs_s, _mat_s, _pat_s, c_hat=0.0, std_dev=0.1)
    assert lower == 0.0


def test_ci_trio_upper_fallback():
    """mcc_ci_trio with c_hat=0.5 falls back to upper_CI=0.5 without raising."""
    mcc = MccEst()
    lower, x, upper = mcc.mcc_ci_trio(_bafs_s, _mat_s, _pat_s, c_hat=0.5, std_dev=0.1)
    assert upper == 0.5


# ---------------------------------------------------------------------------
# Loglik edge cases: boundary c values, all-hom maternal, non-default params
# ---------------------------------------------------------------------------


def test_loglik_trio_c_at_boundaries():
    """loglik_mcc_trio is finite at c=0 and c=0.5."""
    mcc = MccEst()
    for c in [0.0, 0.5]:
        ll = mcc.loglik_mcc_trio(_bafs_s, _mat_s, _pat_s, c=c, std_dev=0.1)
        assert np.isfinite(ll), f"loglik not finite at c={c}"


def test_loglik_poc_c_at_boundaries():
    """loglik_mcc_poc is finite at c=0 and c=0.5."""
    mcc = MccEst()
    for c in [0.0, 0.5]:
        ll = mcc.loglik_mcc_poc(_bafs_s, _mat_s, _freq_s, c=c, std_dev=0.1)
        assert np.isfinite(ll), f"loglik not finite at c={c}"


def test_loglik_trio_all_hom_maternal():
    """loglik_mcc_trio is finite when all maternal sites are homozygous (mg=0 or mg=2)."""
    mcc = MccEst()
    n = 50
    mat_hap_hom = np.vstack([np.ones(n, dtype=np.int32), np.ones(n, dtype=np.int32)])
    pat_hap_het = np.vstack([np.zeros(n, dtype=np.int32), np.ones(n, dtype=np.int32)])
    bafs = np.full(n, 0.75)
    ll = mcc.loglik_mcc_trio(bafs, mat_hap_hom, pat_hap_het, c=0.1, std_dev=0.1)
    assert np.isfinite(ll)


def test_loglik_poc_all_hom_maternal():
    """loglik_mcc_poc is finite when all maternal sites are homozygous."""
    mcc = MccEst()
    n = 50
    mat_hap_hom = np.zeros((2, n), dtype=np.int32)  # all mg=0
    freqs = np.full(n, 0.4)
    bafs = np.full(n, 0.1)
    ll = mcc.loglik_mcc_poc(bafs, mat_hap_hom, freqs, c=0.1, std_dev=0.1)
    assert np.isfinite(ll)


def test_ci_poc_nondefault_alpha():
    """mcc_ci_poc with alpha=0.90 returns a narrower interval than alpha=0.95."""
    mcc = MccEst()
    c_est, std_dev = mcc.est_mcc_poc(_bafs_s, _mat_s, _freq_s)
    lo90, _, hi90 = mcc.mcc_ci_poc(
        _bafs_s, _mat_s, _freq_s, c_hat=c_est, std_dev=std_dev, alpha=0.90
    )
    lo95, _, hi95 = mcc.mcc_ci_poc(
        _bafs_s, _mat_s, _freq_s, c_hat=c_est, std_dev=std_dev, alpha=0.95
    )
    assert (hi90 - lo90) <= (hi95 - lo95)


def test_ci_trio_nondefault_alpha():
    """mcc_ci_trio with alpha=0.90 returns a narrower interval than alpha=0.95."""
    mcc = MccEst()
    c_est, std_dev = mcc.est_mcc_trio(_bafs_s, _mat_s, _pat_s)
    lo90, _, hi90 = mcc.mcc_ci_trio(
        _bafs_s, _mat_s, _pat_s, c_hat=c_est, std_dev=std_dev, alpha=0.90
    )
    lo95, _, hi95 = mcc.mcc_ci_trio(
        _bafs_s, _mat_s, _pat_s, c_hat=c_est, std_dev=std_dev, alpha=0.95
    )
    assert (hi90 - lo90) <= (hi95 - lo95)


def test_est_mcc_poc_nondefault_algo():
    """est_mcc_poc with algo='L-BFGS-B' returns a valid estimate."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_poc(_bafs_s, _mat_s, _freq_s, algo="L-BFGS-B")
    assert 0.0 <= c_est <= 0.5
    assert s_est > 0.0


def test_est_mcc_trio_nondefault_algo():
    """est_mcc_trio with algo='L-BFGS-B' returns a valid estimate."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_trio(_bafs_s, _mat_s, _pat_s, algo="L-BFGS-B")
    assert 0.0 <= c_est <= 0.5
    assert s_est > 0.0


# ---------------------------------------------------------------------------
# Input-validation tests (assert guards)
# ---------------------------------------------------------------------------


def test_loglik_trio_invalid_c_too_large():
    """loglik_mcc_trio raises on c > 0.5."""
    mcc = MccEst()
    try:
        mcc.loglik_mcc_trio(_bafs_s, _mat_s, _pat_s, c=0.6, std_dev=0.1)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


def test_loglik_trio_invalid_std_dev():
    """loglik_mcc_trio raises on std_dev=0."""
    mcc = MccEst()
    try:
        mcc.loglik_mcc_trio(_bafs_s, _mat_s, _pat_s, c=0.1, std_dev=0.0)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


def test_loglik_poc_invalid_freq():
    """loglik_mcc_poc raises on freq > 1."""
    mcc = MccEst()
    bad_freqs = np.full(_bafs_s.size, 1.5)
    try:
        mcc.loglik_mcc_poc(_bafs_s, _mat_s, bad_freqs, c=0.1, std_dev=0.1)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


def test_realistic_ci_trio(m=10000, n=5, c=0.1):
    """Simulate n realistic chromosomes with a high-degree of contamination."""
    pgt_sim = PGTSim()
    data = [
        pgt_sim.full_ploidy_sim(m=m, mix_prop=0.00, std_dev=0.1, seed=i + 1)
        for i in range(n)
    ]
    cc_bafs = np.hstack(
        [
            pgt_sim.sim_cell_contamination(
                baf=data[i]["baf"], haps=data[i]["mat_haps"], fraction=c, seed=i + 1
            )
            for i in range(n)
        ]
    )
    mat_haps = np.hstack([data[i]["mat_haps"] for i in range(n)])
    pat_haps = np.hstack([data[i]["pat_haps"] for i in range(n)])
    mcc = MccEst()
    (c_est, std_dev) = mcc.est_mcc_trio(
        bafs=cc_bafs, mat_haps=mat_haps, pat_haps=pat_haps
    )
    (lower95_c, x, upper95_c) = mcc.mcc_ci_trio(
        bafs=cc_bafs, mat_haps=mat_haps, pat_haps=pat_haps, c_hat=c_est, std_dev=std_dev
    )
    assert x == c_est
    assert lower95_c <= upper95_c

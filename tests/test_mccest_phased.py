"""Tests for phase-aware MCC estimator methods.

Key design notes:
- The phased HMM tracks which maternal haplotype was transmitted to the POC via a
  2-state recombination HMM.  The CI-width advantage over the unphased model is
  real but modest (~3-5 %) and only manifests when:
    (a) maternal sites are heterozygous (so phase matters), and
    (b) consecutive sites share an LD block (so the HMM can resolve the phase state),
    (c) contamination c is comparable to or smaller than BAF noise sigma (so the two
        emission components overlap).
  PGTSim haplotypes have essentially no LD (mean run-length ~2), so those are used
  only for basic sanity checks, not for the CI-comparison assertion.
"""

import numpy as np
import pytest
from karyohmm import MccEst, PGTSim

# ---------------------------------------------------------------------------
# Module-level fixtures (generated once, shared across tests)
# ---------------------------------------------------------------------------

pgt_sim = PGTSim()
_data = pgt_sim.full_ploidy_sim(m=2000, mix_prop=0.0, std_dev=0.1, seed=42)
_c_true = 0.10
_cc_bafs = pgt_sim.sim_cell_contamination(
    baf=_data["baf"],
    haps=_data["mat_haps"],
    fraction=_c_true,
    seed=42,
)
_mat_haps = _data["mat_haps"]
_pat_haps = _data["pat_haps"]
_freqs = _data["af"]

# Build a monotone positions array for the fixture data
_rng_pos = np.random.default_rng(0)
_pos = np.sort(_rng_pos.integers(1, 50_000_000, _data["baf"].size).astype(float))


def _make_ld_block_data(n=4000, c_true=0.08, std_dev=0.12, block_len=400, seed=11):
    """Synthetic trio data drawn directly from the phase-aware emission model.

    Creates alternating LD blocks (length `block_len`) so the HMM can reliably
    resolve which haplotype the POC inherited.  The fetal is always 'inherited
    hap-0'; BAF is drawn from the exact phased emission (not sim_cell_contamination).
    Paternal is all-AA (pg=0) so each het-maternal site is maximally informative
    and consistent with the trio model.
    """
    rng = np.random.default_rng(seed)
    mat_haps = np.zeros((2, n), dtype=np.int32)
    phase = 0
    for start in range(0, n, block_len):
        end = min(start + block_len, n)
        mat_haps[0, start:end] = phase
        mat_haps[1, start:end] = 1 - phase
        phase = 1 - phase
    pat_haps = np.zeros((2, n), dtype=np.int32)  # AA paternal everywhere
    pos = np.linspace(1, 150_000_000, n)
    freqs = np.full(n, 0.5)
    m_h = mat_haps[0, :]  # fetal inherited hap-0
    # Phased emission: m_h=1,pg=0 → N(0.5+c/2, σ²);  m_h=0,pg=0 → N(0, σ²)
    mus = np.where(m_h == 1, 0.5 + c_true / 2, 0.0)
    bafs = np.clip(rng.normal(mus, std_dev), 0.0, 1.0)
    return bafs, mat_haps, pat_haps, freqs, pos, c_true


def _make_ld_block_poc_data(
    n=4000, c_true=0.08, std_dev=0.12, block_len=400, freq=0.05, seed=11
):
    """Synthetic POC data drawn from the phase-aware emission model under HWE.

    Uses a low paternal allele frequency (default 0.05) so the HWE distribution
    is dominated by AA paternal genotypes (prob ≈ 0.90), making the contamination
    signal clearly detectable after marginalising over pg.  BAFs are drawn from
    the exact phased emission for the sampled paternal genotype at each site.
    """
    rng = np.random.default_rng(seed)
    mat_haps = np.zeros((2, n), dtype=np.int32)
    phase = 0
    for start in range(0, n, block_len):
        end = min(start + block_len, n)
        mat_haps[0, start:end] = phase
        mat_haps[1, start:end] = 1 - phase
        phase = 1 - phase
    pos = np.linspace(1, 150_000_000, n)
    freqs = np.full(n, freq)
    m_h = mat_haps[0, :]
    p0, p1, p2 = (1 - freq) ** 2, 2 * freq * (1 - freq), freq**2
    pg = rng.choice([0, 1, 2], size=n, p=[p0, p1, p2])
    mus = np.zeros(n)
    for i in range(n):
        mhi, pgi = int(m_h[i]), int(pg[i])
        if pgi == 0:
            mus[i] = 0.5 + c_true / 2 if mhi == 1 else 0.0
        elif pgi == 1:
            mus[i] = 0.5 if mhi == 1 else rng.choice([c_true / 2, 0.5 - c_true / 2])
        else:
            mus[i] = 1.0 - c_true / 2 if mhi == 1 else 0.5 + c_true / 2
    bafs = np.clip(rng.normal(mus, std_dev), 0.0, 1.0)
    return bafs, mat_haps, freqs, pos, c_true


# ---------------------------------------------------------------------------
# Basic finite-value and bounds tests
# ---------------------------------------------------------------------------


def test_loglik_phased_trio_finite():
    """loglik_mcc_phased_trio returns a finite scalar."""
    mcc = MccEst()
    mat_haps = np.array([[0, 1], [1, 0]], dtype=np.int32)
    pat_haps = np.array([[1, 0], [0, 1]], dtype=np.int32)
    bafs = np.array([0.6, 0.4])
    pos = np.array([1000.0, 2000.0])
    ll = mcc.loglik_mcc_phased_trio(bafs, mat_haps, pat_haps, pos, c=0.1, std_dev=0.1)
    assert np.isfinite(ll)


def test_loglik_phased_poc_finite():
    """loglik_mcc_phased_poc returns a finite scalar."""
    mcc = MccEst()
    mat_haps = np.array([[0, 1], [1, 0]], dtype=np.int32)
    freqs = np.array([0.3, 0.4])
    bafs = np.array([0.6, 0.4])
    pos = np.array([1000.0, 2000.0])
    ll = mcc.loglik_mcc_phased_poc(bafs, mat_haps, freqs, pos, c=0.1, std_dev=0.1)
    assert np.isfinite(ll)


def test_loglik_phased_trio_all_hom_matches_unphased():
    """With all homozygous maternal sites the phased and unphased likelihoods agree.

    Phase is irrelevant when mg in {0,2} so the HMM forward probability equals
    the product of independent site likelihoods from loglik_mcc_trio.
    """
    mcc = MccEst()
    mat_haps = np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.int32)
    pat_haps = np.array([[1, 0, 1, 0], [1, 0, 1, 0]], dtype=np.int32)
    bafs = np.array([0.5, 0.1, 0.9, 0.5])
    pos = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    ll_phased = mcc.loglik_mcc_phased_trio(
        bafs, mat_haps, pat_haps, pos, c=0.1, std_dev=0.1
    )
    ll_unphased = mcc.loglik_mcc_trio(bafs, mat_haps, pat_haps, c=0.1, std_dev=0.1)
    assert np.isclose(ll_phased, ll_unphased, atol=1e-5)


def test_loglik_phased_trio_increases_with_contamination():
    """Phased trio log-likelihood is higher at c_true than at c=0 for contaminated data."""
    mcc = MccEst()
    bafs, mat_haps, pat_haps, _, pos, c_true = _make_ld_block_data()
    ll_true = mcc.loglik_mcc_phased_trio(bafs, mat_haps, pat_haps, pos, c=c_true)
    ll_zero = mcc.loglik_mcc_phased_trio(bafs, mat_haps, pat_haps, pos, c=0.0)
    assert ll_true > ll_zero


def test_loglik_phased_poc_increases_with_contamination():
    """Phased POC log-likelihood is higher at c_true than at c=0 for contaminated data."""
    mcc = MccEst()
    bafs, mat_haps, freqs, pos, c_true = _make_ld_block_poc_data()
    ll_true = mcc.loglik_mcc_phased_poc(bafs, mat_haps, freqs, pos, c=c_true)
    ll_zero = mcc.loglik_mcc_phased_poc(bafs, mat_haps, freqs, pos, c=0.0)
    assert ll_true > ll_zero


# ---------------------------------------------------------------------------
# MLE estimation tests
# ---------------------------------------------------------------------------


def test_mle_phased_trio_bounds():
    """Phased trio MLE c estimate is within [0, 0.5] and sigma > 0."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_phased_trio(_cc_bafs, _mat_haps, _pat_haps, _pos)
    assert 0.0 <= c_est <= 0.5
    assert s_est > 0.0


def test_mle_phased_poc_bounds():
    """Phased POC MLE c estimate is within [0, 0.5] and sigma > 0."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_phased_poc(_cc_bafs, _mat_haps, _freqs, _pos)
    assert 0.0 <= c_est <= 0.5
    assert s_est > 0.0


def test_mle_phased_poc_accuracy():
    """Phased POC MLE recovers c on LD-block data within absolute error 0.04."""
    mcc = MccEst()
    bafs, mat_haps, freqs, pos, c_true = _make_ld_block_poc_data(c_true=0.10)
    c_est, _ = mcc.est_mcc_phased_poc(bafs, mat_haps, freqs, pos)
    assert abs(c_est - c_true) < 0.04


def test_mle_phased_trio_accuracy():
    """Phased trio MLE recovers c on LD-block data within absolute error 0.04."""
    mcc = MccEst()
    bafs, mat_haps, pat_haps, _, pos, c_true = _make_ld_block_data(c_true=0.10)
    c_est, _ = mcc.est_mcc_phased_trio(bafs, mat_haps, pat_haps, pos)
    assert abs(c_est - c_true) < 0.04


# ---------------------------------------------------------------------------
# Confidence interval shape tests
# ---------------------------------------------------------------------------


def test_ci_phased_trio_wellformed():
    """Phased trio CI satisfies lower <= estimate <= upper."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_phased_trio(_cc_bafs, _mat_haps, _pat_haps, _pos)
    lower, x, upper = mcc.mcc_ci_phased_trio(
        _cc_bafs, _mat_haps, _pat_haps, _pos, c_hat=c_est, std_dev=s_est
    )
    assert lower <= x <= upper


def test_ci_phased_poc_wellformed():
    """Phased POC CI satisfies lower <= estimate <= upper."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_phased_poc(_cc_bafs, _mat_haps, _freqs, _pos)
    lower, x, upper = mcc.mcc_ci_phased_poc(
        _cc_bafs, _mat_haps, _freqs, _pos, c_hat=c_est, std_dev=s_est
    )
    assert lower <= x <= upper


def test_ci_phased_trio_covers_truth():
    """Phased trio 95% CI covers the true contamination fraction on LD-block data."""
    mcc = MccEst()
    bafs, mat_haps, pat_haps, _, pos, c_true = _make_ld_block_data(c_true=0.10)
    c_est, s_est = mcc.est_mcc_phased_trio(bafs, mat_haps, pat_haps, pos)
    lower, _, upper = mcc.mcc_ci_phased_trio(
        bafs, mat_haps, pat_haps, pos, c_hat=c_est, std_dev=s_est
    )
    assert lower <= c_true <= upper


def test_ci_phased_poc_covers_truth():
    """Phased POC 95% CI covers the true contamination fraction on LD-block data."""
    mcc = MccEst()
    bafs, mat_haps, freqs, pos, c_true = _make_ld_block_poc_data(c_true=0.10)
    c_est, s_est = mcc.est_mcc_phased_poc(bafs, mat_haps, freqs, pos)
    lower, _, upper = mcc.mcc_ci_phased_poc(
        bafs, mat_haps, freqs, pos, c_hat=c_est, std_dev=s_est
    )
    assert lower <= c_true <= upper


# ---------------------------------------------------------------------------
# CI-width comparison: phased vs unphased
# ---------------------------------------------------------------------------


def test_phased_ci_narrower_than_unphased_trio():
    """On LD-structured data the phased trio CI is narrower than the unphased trio CI.

    The improvement (~3-5 %) arises because the HMM resolves which maternal haplotype
    the POC inherited, converting the per-site 50/50 mixture emission at het-maternal
    sites into a single Gaussian once the phase state is pinned by the LD block.
    This advantage requires c <= sigma; here c=0.08, sigma=0.12.
    """
    mcc = MccEst()
    bafs, mat_haps, pat_haps, _, pos, c_true = _make_ld_block_data(
        c_true=0.08, std_dev=0.12, seed=11
    )
    c_p, s_p = mcc.est_mcc_phased_trio(bafs, mat_haps, pat_haps, pos)
    lo_p, _, hi_p = mcc.mcc_ci_phased_trio(
        bafs, mat_haps, pat_haps, pos, c_hat=c_p, std_dev=s_p
    )
    c_u, s_u = mcc.est_mcc_trio(bafs, mat_haps, pat_haps)
    lo_u, _, hi_u = mcc.mcc_ci_trio(bafs, mat_haps, pat_haps, c_hat=c_u, std_dev=s_u)
    width_phased = hi_p - lo_p
    width_unphased = hi_u - lo_u
    assert width_phased < width_unphased, (
        f"Expected phased ({width_phased:.5f}) < unphased ({width_unphased:.5f})"
    )


def test_phased_ci_narrower_than_unphased_poc():
    """On LD-structured data the phased POC CI is narrower than the unphased POC CI.

    Uses low paternal allele frequency (freq=0.05, ~90 % AA paternal) so the
    HWE marginalisation retains a strong phase-coherent contamination signal.
    """
    mcc = MccEst()
    bafs, mat_haps, freqs, pos, c_true = _make_ld_block_poc_data(
        c_true=0.08, std_dev=0.12, freq=0.05, seed=11
    )
    c_pp, s_pp = mcc.est_mcc_phased_poc(bafs, mat_haps, freqs, pos)
    lo_pp, _, hi_pp = mcc.mcc_ci_phased_poc(
        bafs, mat_haps, freqs, pos, c_hat=c_pp, std_dev=s_pp
    )
    c_up, s_up = mcc.est_mcc_poc(bafs, mat_haps, freqs)
    lo_up, _, hi_up = mcc.mcc_ci_poc(bafs, mat_haps, freqs, c_hat=c_up, std_dev=s_up)
    width_phased = hi_pp - lo_pp
    width_unphased = hi_up - lo_up
    assert width_phased < width_unphased, (
        f"Expected phased ({width_phased:.5f}) < unphased ({width_unphased:.5f})"
    )


# ---------------------------------------------------------------------------
# CI boundary / fallback tests for phased methods
# (cover the ValueError exception handlers in mcc_ci_phased_{trio,poc})
# ---------------------------------------------------------------------------


def test_ci_phased_trio_lower_fallback():
    """mcc_ci_phased_trio with c_hat=0 falls back to lower_CI=0 without raising."""
    mcc = MccEst()
    lower, x, upper = mcc.mcc_ci_phased_trio(
        _cc_bafs, _mat_haps, _pat_haps, _pos, c_hat=0.0, std_dev=0.1
    )
    assert lower == 0.0
    assert x == 0.0
    assert upper >= 0.0


def test_ci_phased_trio_upper_fallback():
    """mcc_ci_phased_trio with c_hat=0.5 falls back to upper_CI=0.5 without raising."""
    mcc = MccEst()
    lower, x, upper = mcc.mcc_ci_phased_trio(
        _cc_bafs, _mat_haps, _pat_haps, _pos, c_hat=0.5, std_dev=0.1
    )
    assert upper == 0.5
    assert x == 0.5


def test_ci_phased_poc_lower_fallback():
    """mcc_ci_phased_poc with c_hat=0 falls back to lower_CI=0 without raising."""
    mcc = MccEst()
    lower, x, upper = mcc.mcc_ci_phased_poc(
        _cc_bafs, _mat_haps, _freqs, _pos, c_hat=0.0, std_dev=0.1
    )
    assert lower == 0.0
    assert x == 0.0


def test_ci_phased_poc_upper_fallback():
    """mcc_ci_phased_poc with c_hat=0.5 falls back to upper_CI=0.5 without raising."""
    mcc = MccEst()
    lower, x, upper = mcc.mcc_ci_phased_poc(
        _cc_bafs, _mat_haps, _freqs, _pos, c_hat=0.5, std_dev=0.1
    )
    assert upper == 0.5
    assert x == 0.5


# ---------------------------------------------------------------------------
# Loglik edge cases: boundary c values, non-default r
# ---------------------------------------------------------------------------


def test_loglik_phased_trio_c_at_boundaries():
    """loglik_mcc_phased_trio is finite at c=0 and c=0.5."""
    mcc = MccEst()
    for c in [0.0, 0.5]:
        ll = mcc.loglik_mcc_phased_trio(_cc_bafs, _mat_haps, _pat_haps, _pos, c=c)
        assert np.isfinite(ll), f"loglik not finite at c={c}"


def test_loglik_phased_poc_c_at_boundaries():
    """loglik_mcc_phased_poc is finite at c=0 and c=0.5."""
    mcc = MccEst()
    for c in [0.0, 0.5]:
        ll = mcc.loglik_mcc_phased_poc(_cc_bafs, _mat_haps, _freqs, _pos, c=c)
        assert np.isfinite(ll), f"loglik not finite at c={c}"


def test_loglik_phased_trio_nondefault_r():
    """loglik_mcc_phased_trio changes with the recombination rate parameter r."""
    mcc = MccEst()
    bafs, mat_haps, pat_haps, _, pos, _ = _make_ld_block_data()
    ll_low_r = mcc.loglik_mcc_phased_trio(bafs, mat_haps, pat_haps, pos, c=0.05, r=1e-9)
    ll_high_r = mcc.loglik_mcc_phased_trio(
        bafs, mat_haps, pat_haps, pos, c=0.05, r=1e-6
    )
    assert ll_low_r != ll_high_r


def test_loglik_phased_poc_nondefault_r():
    """loglik_mcc_phased_poc changes with the recombination rate parameter r."""
    mcc = MccEst()
    bafs, mat_haps, freqs, pos, _ = _make_ld_block_poc_data()
    ll_low_r = mcc.loglik_mcc_phased_poc(bafs, mat_haps, freqs, pos, c=0.05, r=1e-9)
    ll_high_r = mcc.loglik_mcc_phased_poc(bafs, mat_haps, freqs, pos, c=0.05, r=1e-6)
    assert ll_low_r != ll_high_r


def test_ci_phased_poc_nondefault_alpha():
    """mcc_ci_phased_poc with alpha=0.90 is narrower than alpha=0.95."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_phased_poc(_cc_bafs, _mat_haps, _freqs, _pos)
    lo90, _, hi90 = mcc.mcc_ci_phased_poc(
        _cc_bafs, _mat_haps, _freqs, _pos, c_hat=c_est, std_dev=s_est, alpha=0.90
    )
    lo95, _, hi95 = mcc.mcc_ci_phased_poc(
        _cc_bafs, _mat_haps, _freqs, _pos, c_hat=c_est, std_dev=s_est, alpha=0.95
    )
    assert (hi90 - lo90) <= (hi95 - lo95)


def test_ci_phased_trio_nondefault_alpha():
    """mcc_ci_phased_trio with alpha=0.90 is narrower than alpha=0.95."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_phased_trio(_cc_bafs, _mat_haps, _pat_haps, _pos)
    lo90, _, hi90 = mcc.mcc_ci_phased_trio(
        _cc_bafs, _mat_haps, _pat_haps, _pos, c_hat=c_est, std_dev=s_est, alpha=0.90
    )
    lo95, _, hi95 = mcc.mcc_ci_phased_trio(
        _cc_bafs, _mat_haps, _pat_haps, _pos, c_hat=c_est, std_dev=s_est, alpha=0.95
    )
    assert (hi90 - lo90) <= (hi95 - lo95)


# ---------------------------------------------------------------------------
# Input-validation tests (assert guards on phased methods)
# ---------------------------------------------------------------------------


def test_loglik_phased_trio_invalid_c():
    """loglik_mcc_phased_trio raises on c > 0.5."""
    mcc = MccEst()
    try:
        mcc.loglik_mcc_phased_trio(_cc_bafs, _mat_haps, _pat_haps, _pos, c=0.6)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


def test_loglik_phased_poc_invalid_r():
    """loglik_mcc_phased_poc raises on r=0."""
    mcc = MccEst()
    try:
        mcc.loglik_mcc_phased_poc(_cc_bafs, _mat_haps, _freqs, _pos, r=0.0)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


def test_loglik_phased_trio_unsorted_pos():
    """loglik_mcc_phased_trio raises when pos is not strictly increasing."""
    mcc = MccEst()
    bad_pos = _pos.copy()
    bad_pos[10] = bad_pos[9]  # duplicate position
    try:
        mcc.loglik_mcc_phased_trio(_cc_bafs, _mat_haps, _pat_haps, bad_pos)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass


# ---------------------------------------------------------------------------
# Multi-chromosome (genome-wide) phased estimation tests
# ---------------------------------------------------------------------------

_n_chroms_p = 3
_pgt_p = PGTSim()
_chrom_data_p = [
    _pgt_p.full_ploidy_sim(m=600, mix_prop=0.0, std_dev=0.1, seed=300 + i)
    for i in range(_n_chroms_p)
]
_baf_list_p = [d["baf"] for d in _chrom_data_p]
_mat_haps_list_p = [d["mat_haps"] for d in _chrom_data_p]
_pat_haps_list_p = [d["pat_haps"] for d in _chrom_data_p]
_freqs_list_p = [d["af"] for d in _chrom_data_p]

_rng_pos_p = np.random.default_rng(77)
_pos_list_p = [
    np.sort(
        _rng_pos_p.integers(1, 50_000_000, _chrom_data_p[i]["baf"].size).astype(float)
    )
    for i in range(_n_chroms_p)
]

_c_true_genome_p = 0.1
_cc_baf_list_p = [
    _pgt_p.sim_cell_contamination(
        baf=_chrom_data_p[i]["baf"],
        haps=_chrom_data_p[i]["mat_haps"],
        fraction=_c_true_genome_p,
        seed=400 + i,
    )
    for i in range(_n_chroms_p)
]


def test_loglik_genome_phased_poc_finite():
    """loglik_mcc_genome_phased_poc returns a finite scalar."""
    mcc = MccEst()
    ll = mcc.loglik_mcc_genome_phased_poc(
        _baf_list_p, _mat_haps_list_p, _freqs_list_p, _pos_list_p, c=0.1, std_dev=0.1
    )
    assert np.isfinite(ll)


def test_loglik_genome_phased_trio_finite():
    """loglik_mcc_genome_phased_trio returns a finite scalar."""
    mcc = MccEst()
    ll = mcc.loglik_mcc_genome_phased_trio(
        _baf_list_p, _mat_haps_list_p, _pat_haps_list_p, _pos_list_p, c=0.1, std_dev=0.1
    )
    assert np.isfinite(ll)


def test_loglik_genome_phased_poc_equals_sum():
    """loglik_mcc_genome_phased_poc equals the sum of per-chromosome phased POC logliks."""
    mcc = MccEst()
    c, std_dev = 0.08, 0.12
    ll_genome = mcc.loglik_mcc_genome_phased_poc(
        _baf_list_p, _mat_haps_list_p, _freqs_list_p, _pos_list_p, c=c, std_dev=std_dev
    )
    ll_sum = sum(
        mcc.loglik_mcc_phased_poc(b, m, f, pos, c=c, std_dev=std_dev)
        for b, m, f, pos in zip(_baf_list_p, _mat_haps_list_p, _freqs_list_p, _pos_list_p)
    )
    assert np.isclose(ll_genome, ll_sum)


def test_loglik_genome_phased_trio_equals_sum():
    """loglik_mcc_genome_phased_trio equals the sum of per-chromosome phased trio logliks."""
    mcc = MccEst()
    c, std_dev = 0.08, 0.12
    ll_genome = mcc.loglik_mcc_genome_phased_trio(
        _baf_list_p, _mat_haps_list_p, _pat_haps_list_p, _pos_list_p, c=c, std_dev=std_dev
    )
    ll_sum = sum(
        mcc.loglik_mcc_phased_trio(b, m, p, pos, c=c, std_dev=std_dev)
        for b, m, p, pos in zip(
            _baf_list_p, _mat_haps_list_p, _pat_haps_list_p, _pos_list_p
        )
    )
    assert np.isclose(ll_genome, ll_sum)


def test_est_mcc_genome_phased_poc_bounds():
    """est_mcc_genome_phased_poc returns c in [0, 0.5] and sigma > 0."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_genome_phased_poc(
        _cc_baf_list_p, _mat_haps_list_p, _freqs_list_p, _pos_list_p
    )
    assert 0.0 <= c_est <= 0.5
    assert s_est > 0.0


def test_est_mcc_genome_phased_trio_bounds():
    """est_mcc_genome_phased_trio returns c in [0, 0.5] and sigma > 0."""
    mcc = MccEst()
    c_est, s_est = mcc.est_mcc_genome_phased_trio(
        _cc_baf_list_p, _mat_haps_list_p, _pat_haps_list_p, _pos_list_p
    )
    assert 0.0 <= c_est <= 0.5
    assert s_est > 0.0


def test_est_mcc_per_chrom_phased_poc_length_and_bounds():
    """est_mcc_per_chrom_phased_poc returns one valid (c, sigma) per chromosome."""
    mcc = MccEst()
    results = mcc.est_mcc_per_chrom_phased_poc(
        _cc_baf_list_p, _mat_haps_list_p, _freqs_list_p, _pos_list_p
    )
    assert len(results) == _n_chroms_p
    for c_est, s_est in results:
        assert 0.0 <= c_est <= 0.5
        assert s_est > 0.0


def test_est_mcc_per_chrom_phased_trio_length_and_bounds():
    """est_mcc_per_chrom_phased_trio returns one valid (c, sigma) per chromosome."""
    mcc = MccEst()
    results = mcc.est_mcc_per_chrom_phased_trio(
        _cc_baf_list_p, _mat_haps_list_p, _pat_haps_list_p, _pos_list_p
    )
    assert len(results) == _n_chroms_p
    for c_est, s_est in results:
        assert 0.0 <= c_est <= 0.5
        assert s_est > 0.0

"""Integrated tests for genome-wide and per-chromosome MCC estimation.

Two fixture types are used throughout:

- **PGTSim fixture** (``_cc_baf_list`` etc.): three chromosomes from
  ``full_ploidy_sim`` with contamination added via ``sim_cell_contamination``.
  Used for detection and per-chromosome consistency tests.  Note that
  ``sim_cell_contamination(fraction=f)`` produces a BAF shift of roughly ``f/2``
  at informative sites, which the MLE recovers as a model-c of ≈ f/2.  Tests
  therefore use the MLE-based LRT rather than asserting ll(fraction) > ll(0).

- **Model-consistent fixture** (``_model_*`` prefixed): BAF drawn directly from
  the MCC emission model so accuracy tests can assert abs(c_est - c_true) < 0.04.
  These follow the same LD-block approach used in test_mccest_phased.py.
"""

import numpy as np
import pytest
from karyohmm import MccEst, PGTSim

# ---------------------------------------------------------------------------
# Fixture 1 — PGTSim with sim_cell_contamination (3 chromosomes, 2 000 SNPs)
# ---------------------------------------------------------------------------

N_CHROMS = 3
M_PER_CHROM = 2000
FRACTION = 0.10   # sim_cell_contamination fraction; MCC model sees ≈ 0.05

_sim = PGTSim()
_chrom_data = [
    _sim.full_ploidy_sim(m=M_PER_CHROM, mix_prop=0.0, std_dev=0.10, seed=700 + i)
    for i in range(N_CHROMS)
]

_baf_list = [d["baf"] for d in _chrom_data]
_mat_haps_list = [d["mat_haps"] for d in _chrom_data]
_pat_haps_list = [d["pat_haps"] for d in _chrom_data]
_freqs_list = [d["af"] for d in _chrom_data]
_pos_list = [d["pos"] for d in _chrom_data]  # positions from full_ploidy_sim

_cc_baf_list = [
    _sim.sim_cell_contamination(
        baf=_chrom_data[i]["baf"],
        haps=_chrom_data[i]["mat_haps"],
        fraction=FRACTION,
        seed=800 + i,
    )
    for i in range(N_CHROMS)
]


# ---------------------------------------------------------------------------
# Fixture 2 — BAF drawn directly from MCC emission model (LD-block structure)
# ---------------------------------------------------------------------------

def _make_genome_ld_trio(n_chroms=3, m=2000, c_true=0.10, std_dev=0.12,
                          block_len=400, seed=11):
    """Multi-chromosome trio data sampled from the phase-aware MCC emission.

    Maternal haplotypes form alternating LD blocks so the HMM can resolve
    the transmitted allele.  Paternal is all-AA (pg=0) for maximum signal.
    """
    baf_list, mat_list, pat_list, freqs_list, pos_list = [], [], [], [], []
    for i in range(n_chroms):
        rng = np.random.default_rng(seed + i * 100)
        mat_haps = np.zeros((2, m), dtype=np.int32)
        phase = 0
        for start in range(0, m, block_len):
            end = min(start + block_len, m)
            mat_haps[0, start:end] = phase
            mat_haps[1, start:end] = 1 - phase
            phase = 1 - phase
        pat_haps = np.zeros((2, m), dtype=np.int32)
        pos = np.linspace(1 + i * 200_000_000, (i + 1) * 200_000_000, m)
        freqs = np.full(m, 0.5)
        m_h = mat_haps[0, :]
        mus = np.where(m_h == 1, 0.5 + c_true / 2, 0.0)
        bafs = np.clip(rng.normal(mus, std_dev), 0.0, 1.0)
        baf_list.append(bafs)
        mat_list.append(mat_haps)
        pat_list.append(pat_haps)
        freqs_list.append(freqs)
        pos_list.append(pos)
    return baf_list, mat_list, pat_list, freqs_list, pos_list, c_true


def _make_genome_ld_poc(n_chroms=3, m=2000, c_true=0.10, std_dev=0.12,
                         block_len=400, freq=0.05, seed=11):
    """Multi-chromosome POC data sampled from the phase-aware MCC emission.

    Low paternal allele frequency (default 0.05) keeps ~90 % of sites as
    AA paternal, making the contamination signal clearly detectable.
    """
    baf_list, mat_list, freqs_list, pos_list = [], [], [], []
    for i in range(n_chroms):
        rng = np.random.default_rng(seed + i * 100)
        mat_haps = np.zeros((2, m), dtype=np.int32)
        phase = 0
        for start in range(0, m, block_len):
            end = min(start + block_len, m)
            mat_haps[0, start:end] = phase
            mat_haps[1, start:end] = 1 - phase
            phase = 1 - phase
        pos = np.linspace(1 + i * 200_000_000, (i + 1) * 200_000_000, m)
        freqs = np.full(m, freq)
        m_h = mat_haps[0, :]
        p0, p1, p2 = (1 - freq) ** 2, 2 * freq * (1 - freq), freq ** 2
        pg = rng.choice([0, 1, 2], size=m, p=[p0, p1, p2])
        mus = np.zeros(m)
        for j in range(m):
            mhi, pgi = int(m_h[j]), int(pg[j])
            if pgi == 0:
                mus[j] = 0.5 + c_true / 2 if mhi == 1 else 0.0
            elif pgi == 1:
                mus[j] = 0.5 if mhi == 1 else rng.choice([c_true / 2, 0.5 - c_true / 2])
            else:
                mus[j] = 1.0 - c_true / 2 if mhi == 1 else 0.5 + c_true / 2
        bafs = np.clip(rng.normal(mus, std_dev), 0.0, 1.0)
        baf_list.append(bafs)
        mat_list.append(mat_haps)
        freqs_list.append(freqs)
        pos_list.append(pos)
    return baf_list, mat_list, freqs_list, pos_list, c_true


# Pre-compute model-consistent fixtures once at module level
(
    _model_baf_trio, _model_mat_trio, _model_pat_trio,
    _model_freqs_trio, _model_pos_trio, _model_c_trio,
) = _make_genome_ld_trio(seed=11)

(
    _model_baf_poc, _model_mat_poc,
    _model_freqs_poc, _model_pos_poc, _model_c_poc,
) = _make_genome_ld_poc(seed=11)


# ---------------------------------------------------------------------------
# 1. Contamination detection: MLE c_hat > threshold on PGTSim data.
#    Uses the MLE as the alternative hypothesis so the LRT is guaranteed > 0.
# ---------------------------------------------------------------------------


def test_genome_trio_detects_contamination():
    """Unphased trio genome MLE is > 0.01 on contaminated PGTSim data."""
    mcc = MccEst()
    c_hat, _ = mcc.est_mcc_genome_trio(_cc_baf_list, _mat_haps_list, _pat_haps_list)
    assert c_hat > 0.01, f"c_hat = {c_hat:.4f}; expected > 0.01"


def test_genome_poc_detects_contamination():
    """Unphased POC genome MLE is > 0.01 on contaminated PGTSim data."""
    mcc = MccEst()
    c_hat, _ = mcc.est_mcc_genome_poc(_cc_baf_list, _mat_haps_list, _freqs_list)
    assert c_hat > 0.01, f"c_hat = {c_hat:.4f}; expected > 0.01"


def test_genome_phased_trio_detects_contamination():
    """Phase-aware trio genome MLE is > 0.01 on contaminated PGTSim data."""
    mcc = MccEst()
    c_hat, _ = mcc.est_mcc_genome_phased_trio(
        _cc_baf_list, _mat_haps_list, _pat_haps_list, _pos_list
    )
    assert c_hat > 0.01, f"c_hat = {c_hat:.4f}; expected > 0.01"


def test_genome_phased_poc_detects_contamination():
    """Phase-aware POC genome MLE is > 0.01 on contaminated PGTSim data."""
    mcc = MccEst()
    c_hat, _ = mcc.est_mcc_genome_phased_poc(
        _cc_baf_list, _mat_haps_list, _freqs_list, _pos_list
    )
    assert c_hat > 0.01, f"c_hat = {c_hat:.4f}; expected > 0.01"


# ---------------------------------------------------------------------------
# 2. Phase-aware vs unphased comparison: both LRTs must be positive (the
#    contamination signal is non-zero) and the two MLEs must agree to within
#    0.05 (both are estimating the same underlying quantity).
# ---------------------------------------------------------------------------


def test_phased_unphased_lrt_both_positive_trio():
    """Phased and unphased trio LRTs are both positive on contaminated PGTSim data."""
    mcc = MccEst()
    c_u, s_u = mcc.est_mcc_genome_trio(_cc_baf_list, _mat_haps_list, _pat_haps_list)
    lrt_u = 2 * (
        mcc.loglik_mcc_genome_trio(_cc_baf_list, _mat_haps_list, _pat_haps_list, c=c_u, std_dev=s_u)
        - mcc.loglik_mcc_genome_trio(_cc_baf_list, _mat_haps_list, _pat_haps_list, c=0.0, std_dev=s_u)
    )
    c_p, s_p = mcc.est_mcc_genome_phased_trio(
        _cc_baf_list, _mat_haps_list, _pat_haps_list, _pos_list
    )
    lrt_p = 2 * (
        mcc.loglik_mcc_genome_phased_trio(
            _cc_baf_list, _mat_haps_list, _pat_haps_list, _pos_list, c=c_p, std_dev=s_p
        )
        - mcc.loglik_mcc_genome_phased_trio(
            _cc_baf_list, _mat_haps_list, _pat_haps_list, _pos_list, c=0.0, std_dev=s_p
        )
    )
    assert lrt_u > 0, f"Unphased trio LRT = {lrt_u:.2f}"
    assert lrt_p > 0, f"Phased trio LRT = {lrt_p:.2f}"
    assert abs(c_u - c_p) < 0.05, f"Phased ({c_p:.3f}) and unphased ({c_u:.3f}) estimates diverge"


def test_phased_unphased_lrt_both_positive_poc():
    """Phased and unphased POC LRTs are both positive on contaminated PGTSim data."""
    mcc = MccEst()
    c_u, s_u = mcc.est_mcc_genome_poc(_cc_baf_list, _mat_haps_list, _freqs_list)
    lrt_u = 2 * (
        mcc.loglik_mcc_genome_poc(_cc_baf_list, _mat_haps_list, _freqs_list, c=c_u, std_dev=s_u)
        - mcc.loglik_mcc_genome_poc(_cc_baf_list, _mat_haps_list, _freqs_list, c=0.0, std_dev=s_u)
    )
    c_p, s_p = mcc.est_mcc_genome_phased_poc(
        _cc_baf_list, _mat_haps_list, _freqs_list, _pos_list
    )
    lrt_p = 2 * (
        mcc.loglik_mcc_genome_phased_poc(
            _cc_baf_list, _mat_haps_list, _freqs_list, _pos_list, c=c_p, std_dev=s_p
        )
        - mcc.loglik_mcc_genome_phased_poc(
            _cc_baf_list, _mat_haps_list, _freqs_list, _pos_list, c=0.0, std_dev=s_p
        )
    )
    assert lrt_u > 0, f"Unphased POC LRT = {lrt_u:.2f}"
    assert lrt_p > 0, f"Phased POC LRT = {lrt_p:.2f}"
    assert abs(c_u - c_p) < 0.05, f"Phased ({c_p:.3f}) and unphased ({c_u:.3f}) estimates diverge"


# ---------------------------------------------------------------------------
# 3. MLE accuracy on model-consistent data (all four model variants).
#    Three chromosomes of 2 000 SNPs each with LD-block structure; BAF drawn
#    directly from the MCC emission so the MLE should recover c within 0.04.
# ---------------------------------------------------------------------------


def test_genome_mle_trio_accuracy():
    """Genome-wide unphased trio MLE recovers c within 0.04 on model-consistent data."""
    mcc = MccEst()
    c_est, _ = mcc.est_mcc_genome_trio(_model_baf_trio, _model_mat_trio, _model_pat_trio)
    assert abs(c_est - _model_c_trio) < 0.04, f"|{c_est:.3f} - {_model_c_trio}| >= 0.04"


def test_genome_mle_poc_accuracy():
    """Genome-wide unphased POC MLE recovers c within 0.04 on model-consistent data."""
    mcc = MccEst()
    c_est, _ = mcc.est_mcc_genome_poc(_model_baf_poc, _model_mat_poc, _model_freqs_poc)
    assert abs(c_est - _model_c_poc) < 0.04, f"|{c_est:.3f} - {_model_c_poc}| >= 0.04"


def test_genome_phased_mle_trio_accuracy():
    """Genome-wide phased trio MLE recovers c within 0.04 on model-consistent data."""
    mcc = MccEst()
    c_est, _ = mcc.est_mcc_genome_phased_trio(
        _model_baf_trio, _model_mat_trio, _model_pat_trio, _model_pos_trio
    )
    assert abs(c_est - _model_c_trio) < 0.04, f"|{c_est:.3f} - {_model_c_trio}| >= 0.04"


def test_genome_phased_mle_poc_accuracy():
    """Genome-wide phased POC MLE recovers c within 0.04 on model-consistent data."""
    mcc = MccEst()
    c_est, _ = mcc.est_mcc_genome_phased_poc(
        _model_baf_poc, _model_mat_poc, _model_freqs_poc, _model_pos_poc
    )
    assert abs(c_est - _model_c_poc) < 0.04, f"|{c_est:.3f} - {_model_c_poc}| >= 0.04"


# ---------------------------------------------------------------------------
# 4. Per-chromosome estimates are consistent with the genome-wide MLE.
#    All three chromosomes are disomic with identical contamination, so
#    per-chrom estimates should cluster near the genome-wide estimate.
#    This validates the workflow where per-chrom outliers flag aneuploidies.
# ---------------------------------------------------------------------------


def test_per_chrom_trio_consistent_with_genome():
    """Per-chrom unphased trio estimates are within 0.05 of the genome-wide MLE."""
    mcc = MccEst()
    c_genome, _ = mcc.est_mcc_genome_trio(_cc_baf_list, _mat_haps_list, _pat_haps_list)
    per_chrom = mcc.est_mcc_per_chrom_trio(_cc_baf_list, _mat_haps_list, _pat_haps_list)
    for i, (c_chr, _) in enumerate(per_chrom):
        assert abs(c_chr - c_genome) < 0.05, (
            f"Chrom {i}: |{c_chr:.3f} - {c_genome:.3f}| >= 0.05"
        )


def test_per_chrom_poc_consistent_with_genome():
    """Per-chrom unphased POC estimates are within 0.05 of the genome-wide MLE."""
    mcc = MccEst()
    c_genome, _ = mcc.est_mcc_genome_poc(_cc_baf_list, _mat_haps_list, _freqs_list)
    per_chrom = mcc.est_mcc_per_chrom_poc(_cc_baf_list, _mat_haps_list, _freqs_list)
    for i, (c_chr, _) in enumerate(per_chrom):
        assert abs(c_chr - c_genome) < 0.05, (
            f"Chrom {i}: |{c_chr:.3f} - {c_genome:.3f}| >= 0.05"
        )


def test_per_chrom_phased_trio_consistent_with_genome():
    """Per-chrom phased trio estimates are within 0.05 of the genome-wide MLE."""
    mcc = MccEst()
    c_genome, _ = mcc.est_mcc_genome_phased_trio(
        _cc_baf_list, _mat_haps_list, _pat_haps_list, _pos_list
    )
    per_chrom = mcc.est_mcc_per_chrom_phased_trio(
        _cc_baf_list, _mat_haps_list, _pat_haps_list, _pos_list
    )
    for i, (c_chr, _) in enumerate(per_chrom):
        assert abs(c_chr - c_genome) < 0.05, (
            f"Chrom {i}: |{c_chr:.3f} - {c_genome:.3f}| >= 0.05"
        )


# ---------------------------------------------------------------------------
# 5. Concatenation equivalence: the optimised genome-wide unphased loglik
#    (which concatenates arrays into a single call) must equal the naive
#    per-chromosome sum.  Guards against the efficiency refactor breaking
#    correctness.  Phased variants are NOT concatenated (HMM resets per
#    chromosome), so they are not tested here.
# ---------------------------------------------------------------------------


def test_genome_poc_concat_equals_sum():
    """loglik_mcc_genome_poc equals the naive per-chrom sum at all tested (c, sigma) values."""
    mcc = MccEst()
    for c, std_dev in [(0.0, 0.1), (0.05, 0.1), (0.10, 0.15)]:
        ll_genome = mcc.loglik_mcc_genome_poc(
            _baf_list, _mat_haps_list, _freqs_list, c=c, std_dev=std_dev
        )
        ll_sum = sum(
            mcc.loglik_mcc_poc(b, m, f, c=c, std_dev=std_dev)
            for b, m, f in zip(_baf_list, _mat_haps_list, _freqs_list)
        )
        assert np.isclose(ll_genome, ll_sum, rtol=1e-10), (
            f"c={c}: concatenated {ll_genome:.6f} != sum {ll_sum:.6f}"
        )


def test_genome_trio_concat_equals_sum():
    """loglik_mcc_genome_trio equals the naive per-chrom sum at all tested (c, sigma) values."""
    mcc = MccEst()
    for c, std_dev in [(0.0, 0.1), (0.05, 0.1), (0.10, 0.15)]:
        ll_genome = mcc.loglik_mcc_genome_trio(
            _baf_list, _mat_haps_list, _pat_haps_list, c=c, std_dev=std_dev
        )
        ll_sum = sum(
            mcc.loglik_mcc_trio(b, m, p, c=c, std_dev=std_dev)
            for b, m, p in zip(_baf_list, _mat_haps_list, _pat_haps_list)
        )
        assert np.isclose(ll_genome, ll_sum, rtol=1e-10), (
            f"c={c}: concatenated {ll_genome:.6f} != sum {ll_sum:.6f}"
        )

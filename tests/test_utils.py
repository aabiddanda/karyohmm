"""Testing module for utility functions."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from karyohmm_utils import (
    emission_baf,
    emission_baf_mcc,
    emission_depth,
    emission_lrr,
    emission_lrr_mcc,
    emission_reads,
    mat_dosage,
    norm_logl,
    pat_dosage,
    truncnorm_pdf,
)
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar
from scipy.stats import binom, norm, poisson, truncnorm

from karyohmm import PGTSim


@pytest.mark.parametrize(
    "hap,state,expected",
    [
        ([0, 1], (0, -1, 1, -1), 0),
        ([0, 1], (1, -1, 1, -1), 1),
        ([1, 1], (0, 1, 1, -1), 2),
        ([1, 1], (-1, -1, -1, -1), -1),
        # UPD states: total ploidy (k) is 2, but the maternal side of the
        # state tuple is entirely -1 (paternal isodisomy/heterodisomy).
        # The mother contributed nothing, so the dosage must be 0 -- NOT
        # `hap[-1]`, which Python/Cython silently resolves to `hap[1]`.
        ([0, 1], (-1, -1, 0, 0), 0),
        ([0, 1], (-1, -1, 0, 1), 0),
        ([1, 0], (-1, -1, 0, 0), 0),
        # Maternal UPD states: both maternal copies are used and the
        # paternal side is absent -- these should still sum correctly.
        ([1, 0], (0, 1, -1, -1), 1),
        ([1, 1], (0, 0, -1, -1), 2),
    ],
)
def test_mat_dosage(hap, state, expected):
    """Test the maternal dosage function."""
    assert mat_dosage(hap, state) == expected


@pytest.mark.parametrize(
    "hap,state,expected",
    [
        ([0, 1], (0, -1, 1, -1), 1),
        ([0, 1], (1, -1, 0, -1), 0),
        ([0, 1], (-1, -1, 0, -1), 0),
        ([0, 1], (1, -1, 0, 1), 1.0),
        ([1, 1], (1, -1, 0, 1), 2.0),
        ([1, 0], (1, -1, 0, 0), 2.0),
        ([1, 1], (-1, -1, -1, -1), -1),
        # UPD states: total ploidy (k) is 2, but the paternal side of the
        # state tuple is entirely -1 (maternal isodisomy/heterodisomy).
        ([0, 1], (0, 0, -1, -1), 0),
        ([0, 1], (0, 1, -1, -1), 0),
        ([1, 0], (0, 0, -1, -1), 0),
        # Paternal UPD states: both paternal copies are used and the
        # maternal side is absent -- these should still sum correctly.
        ([1, 0], (-1, -1, 0, 1), 1),
        ([1, 1], (-1, -1, 0, 0), 2),
    ],
)
def test_pat_dosage(hap, state, expected):
    """Test the paternal dosage function."""
    assert pat_dosage(hap, state) == expected


@given(
    pi0=st.floats(
        min_value=0, max_value=1, exclude_min=True, exclude_max=True, allow_nan=False
    ),
    sigma=st.floats(
        min_value=1e-2,
        max_value=1.0,
        exclude_min=True,
        exclude_max=True,
        allow_nan=False,
    ),
    m=st.integers(min_value=0, max_value=1),
    p=st.integers(min_value=0, max_value=1),
    k=st.integers(min_value=2, max_value=2),
)
@settings(deadline=None, max_examples=25)
def test_emission_disomy(pi0, sigma, m, p, k):
    """Test that the emission under disomy integrates to 1."""
    bs = np.linspace(0, 1, 1000)
    e = [np.exp(emission_baf(b, m=m, p=p, k=k, pi0=pi0, std_dev=sigma)) for b in bs]
    integrand = trapezoid(e, bs)
    assert np.isclose(integrand, 1, atol=1e-03)


@given(
    pi0=st.floats(
        min_value=0, max_value=1, exclude_min=True, exclude_max=True, allow_nan=False
    ),
    sigma=st.floats(
        min_value=1e-2,
        max_value=1.0,
        exclude_min=True,
        exclude_max=True,
        allow_nan=False,
    ),
    m=st.integers(min_value=0, max_value=1),
)
@settings(deadline=None, max_examples=25)
def test_emission_monosomy(pi0, sigma, m):
    """Test that the emission under disomy integrates to 1."""
    bs = np.linspace(0, 1, 1000)
    e = [np.exp(emission_baf(b, m=m, p=0, k=1, pi0=pi0, std_dev=sigma)) for b in bs]
    integrand = trapezoid(e, bs)
    assert np.isclose(integrand, 1, atol=1e-03)


@given(
    pi0=st.floats(
        min_value=0, max_value=1, exclude_min=True, exclude_max=True, allow_nan=False
    ),
    sigma=st.floats(
        min_value=1e-2,
        max_value=1.0,
        exclude_min=True,
        exclude_max=True,
        allow_nan=False,
    ),
    m=st.integers(min_value=0, max_value=2),
    p=st.integers(min_value=0, max_value=1),
)
@settings(deadline=None, max_examples=25)
def test_emission_trisomy(pi0, sigma, m, p):
    """Test that the emission under disomy integrates to 1."""
    bs = np.linspace(0, 1, 1000)
    e = [np.exp(emission_baf(b, m=m, p=p, k=3, pi0=pi0, std_dev=sigma)) for b in bs]
    integrand = trapezoid(e, bs)
    assert np.isclose(integrand, 1, atol=1e-03)


@given(
    baf=st.floats(min_value=0, max_value=0, allow_nan=False),
    m=st.integers(min_value=0, max_value=1),
    p=st.integers(min_value=0, max_value=1),
    sigma=st.floats(
        min_value=1e-2,
        max_value=1.0,
        exclude_min=True,
        exclude_max=True,
        allow_nan=False,
    ),
    k=st.integers(min_value=2, max_value=3),
)
@settings(deadline=None)
def test_emission_overall(baf, m, p, k, sigma):
    """Test the overall admissions distribution."""
    mu = (m + p) / k
    a, b = 0, 1
    true_logpdf = truncnorm.logpdf(
        baf, (a - mu) / sigma, (b - mu) / sigma, loc=mu, scale=sigma
    )
    baf_log_pdf = emission_baf(baf, m=m, p=p, k=k, pi0=1e-12, std_dev=sigma)
    assert np.isclose(baf_log_pdf, true_logpdf)


@given(
    pi0=st.floats(
        min_value=0, max_value=1, exclude_min=True, exclude_max=True, allow_nan=False
    ),
    sigma=st.floats(
        min_value=1e-2,
        max_value=0.5,
        exclude_min=True,
        exclude_max=True,
        allow_nan=False,
    ),
    nsibs=st.integers(min_value=1, max_value=10),
    switch=st.booleans(),
)
@settings(deadline=None, max_examples=25)
def test_sim_joint_het(switch, pi0, sigma, nsibs):
    """Test the simulation of the joint heterozygotes."""
    true_haps1, true_haps2, haps1, haps2, bafs, genos = PGTSim().sim_joint_het(
        switch=switch, mix_prop=pi0, std_dev=sigma, nsibs=nsibs
    )
    assert true_haps1.ndim == 2
    assert true_haps2.ndim == 2
    assert len(bafs) == nsibs
    # Make sure that the first ones are hets and second are homozyg
    assert np.all(np.sum(true_haps1, axis=0) == 1)
    assert np.all(np.sum(haps1, axis=0) == 1)
    assert np.all(np.sum(true_haps2, axis=0) != 1)
    assert np.all(np.sum(haps2, axis=0) != 1)
    if not switch:
        assert np.all(true_haps1 == haps1)
    else:
        assert ~np.all(true_haps1 == haps1)


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("n", [1, 5, 30])
@pytest.mark.parametrize("eps", [1e-3, 2e-2, 0.1])
def test_emission_reads_matches_binomial(k, n, eps):
    """The read emission is exactly Binomial(depth, q) under the flip model."""
    for dose in range(k + 1):
        mu = dose / k
        q = mu * (1.0 - eps) + (1.0 - mu) * eps
        for alt in range(n + 1):
            obs = emission_reads(alt, n - alt, float(dose), 0.0, k, eps)
            assert obs == pytest.approx(binom.logpmf(alt, n, q))


@pytest.mark.parametrize("eps", [1e-3, 5e-2])
def test_emission_reads_nullisomy(eps):
    """Nullisomy carries no true allele, so every read is background at rate eps."""
    for n, alt in [(5, 0), (5, 1), (20, 3)]:
        obs = emission_reads(alt, n - alt, -1.0, -1.0, 0, eps)
        assert obs == pytest.approx(binom.logpmf(alt, n, eps))


def test_emission_reads_homozygous_is_finite():
    """A discordant read must not annihilate a homozygous state."""
    assert np.isfinite(emission_reads(1, 29, 0.0, 0.0, 2, 1e-2))
    assert np.isfinite(emission_reads(29, 1, 1.0, 1.0, 2, 1e-2))


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("coverage", [1.0, 20.0])
def test_emission_depth_matches_poisson(k, coverage):
    """The depth emission is Poisson in the copy-number-scaled mean."""
    mismap_rate = 1e-2
    lambda_k = coverage * ((k + mismap_rate) / 2.0)
    for depth in [0, 1, 5, 40]:
        obs = emission_depth(depth, k, coverage, mismap_rate)
        assert obs == pytest.approx(poisson.logpmf(depth, lambda_k))


def test_emission_depth_missing_sentinel():
    """Missing depth is flagged with -9, matching the emission_lrr convention."""
    assert emission_depth(-9, 2, 20.0, 1e-2) == -1


@pytest.mark.parametrize("true_ploidy", [0, 1, 2, 3])
def test_emission_depth_recovers_ploidy(true_ploidy):
    """Summed depth log-lik is maximised at the ploidy that generated the data."""
    coverage, mismap_rate = 20.0, 1e-2
    data = PGTSim().full_ploidy_sim(
        m=2000,
        length=10e6,
        ploidy=true_ploidy,
        reads=True,
        coverage=coverage,
        mismap_rate=mismap_rate,
        seed=42,
    )
    depth = data["alt_reads"].astype(int) + data["ref_reads"].astype(int)
    lls = [
        sum(emission_depth(int(n), k, coverage, mismap_rate) for n in depth)
        for k in range(4)
    ]
    assert int(np.argmax(lls)) == true_ploidy


@pytest.mark.parametrize("true_eps", [2e-3, 1e-2, 5e-2])
def test_emission_reads_recovers_eps(true_eps):
    """Maximising the read emission recovers the simulator's error rate."""
    data = PGTSim().full_ploidy_sim(
        m=4000, length=20e6, ploidy=2, reads=True, coverage=30.0, eps=true_eps, seed=42
    )
    alt = data["alt_reads"].astype(int)
    ref = data["ref_reads"].astype(int)
    geno = data["geno_embryo"].astype(float)

    def nll(eps):
        return -sum(
            emission_reads(int(a), int(r), float(g), 0.0, 2, eps)
            for a, r, g in zip(alt, ref, geno)
        )

    res = minimize_scalar(nll, bounds=(1e-5, 0.4), method="bounded")
    assert res.x == pytest.approx(true_eps, rel=0.1)


@pytest.mark.parametrize("true_coverage", [5.0, 20.0, 60.0])
def test_emission_depth_recovers_coverage(true_coverage):
    """Maximising the depth emission recovers the simulator's coverage."""
    data = PGTSim().full_ploidy_sim(
        m=4000,
        length=20e6,
        ploidy=2,
        reads=True,
        coverage=true_coverage,
        mismap_rate=1e-2,
        seed=42,
    )
    depth = data["alt_reads"].astype(int) + data["ref_reads"].astype(int)

    def nll(coverage):
        return -sum(emission_depth(int(n), 2, coverage, 1e-2) for n in depth)

    res = minimize_scalar(nll, bounds=(0.1, 300.0), method="bounded")
    assert res.x == pytest.approx(true_coverage, rel=0.05)


@pytest.mark.parametrize("baf", [0.0, 0.2, 0.5, 0.7, 1.0])
@pytest.mark.parametrize(
    "m,p,k", [(-1, -1, 0), (0, -1, 1), (1, -1, 1), (0, 1, 2), (1, 1, 2), (2, 1, 3)]
)
@pytest.mark.parametrize("mg", [0, 1, 2])
def test_emission_baf_mcc_no_contamination(baf, m, p, k, mg):
    """At f = 0 the contamination-aware BAF emission is exactly emission_baf."""
    assert emission_baf_mcc(baf, m, p, mg, f=0.0, pi0=0.3, std_dev=0.1, k=k) == (
        emission_baf(baf, m, p, pi0=0.3, std_dev=0.1, k=k)
    )


@pytest.mark.parametrize("f", [0.05, 0.2, 0.4])
@pytest.mark.parametrize(
    "m,p,mg,k",
    [
        # Disomic het embryo, mother homozygous B: BAF 0.5 + f / 2.
        (1, 0, 2, 2),
        # Disomic het embryo, mother homozygous A: BAF 0.5 - f / 2.
        (0, 1, 0, 2),
        # Monosomy, trisomy, and nullisomy, where copy weighting matters.
        (0, -1, 2, 1),
        (1, 1, 2, 3),
        (2, 0, 1, 3),
        (-1, -1, 1, 0),
    ],
)
def test_emission_baf_mcc_mean(f, m, p, mg, k):
    """The emission is a truncated normal at the copy-weighted mixture mean."""
    d = 0 if k == 0 else m + p
    mu = ((1 - f) * d + f * mg) / ((1 - f) * k + 2 * f)
    for baf in [0.3, 0.55, 0.8]:
        assert np.isclose(
            emission_baf_mcc(baf, m, p, mg, f=f, std_dev=0.1, k=k),
            truncnorm_pdf(baf, 0.0, 1.0, mu=mu, sigma=0.1),
        )
    if (k == 2) and (d == 1):
        assert np.isclose(mu, 0.5 + f / 2 if mg == 2 else 0.5 - f / 2)


@pytest.mark.parametrize("f", [0.05, 0.3])
def test_emission_baf_mcc_point_masses(f):
    """Point masses at 0 and 1 apply only when every cell is homozygous for that allele."""
    # Embryo and mother both homozygous A: the mixture is exactly 0.
    shared = emission_baf_mcc(0.0, 0, 0, 0, f=f, pi0=0.5, std_dev=0.1, k=2)
    # Embryo homozygous A but the mother is het: no point mass at 0.
    mixed = emission_baf_mcc(0.0, 0, 0, 1, f=f, pi0=0.5, std_dev=0.1, k=2)
    mu = f * 1 / 2
    assert np.isclose(mixed, truncnorm_pdf(0.0, 0.0, 1.0, mu=mu, sigma=0.1))
    assert shared > mixed
    # The same at 1, for a trisomic embryo homozygous B and a homozygous B mother.
    assert emission_baf_mcc(1.0, 2, 1, 2, f=f, pi0=0.5, std_dev=0.1, k=3) > (
        emission_baf_mcc(1.0, 2, 1, 1, f=f, pi0=0.5, std_dev=0.1, k=3)
    )


@pytest.mark.parametrize("f", [0.1, 0.3])
@pytest.mark.parametrize("mg,k", [(0, 2), (1, 2), (2, 3), (1, 1)])
def test_emission_baf_mcc_integrates_to_one(f, mg, k):
    """Away from the point masses the BAF emission is a density on [0, 1]."""
    x = np.linspace(0, 1, 20001)
    dens = np.exp([emission_baf_mcc(b, 1, 0, mg, f=f, std_dev=0.1, k=k) for b in x])
    assert np.isclose(trapezoid(dens, x), 1.0, atol=1e-3)


@pytest.mark.parametrize("lrr", [-9.0, -2.0, -0.4, 0.0, 0.3])
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_emission_lrr_mcc_no_contamination(lrr, k):
    """At f = 0 and delta = 0 the contamination-aware LRR emission is exactly emission_lrr."""
    assert emission_lrr_mcc(lrr, k=k, std_dev=0.2) == emission_lrr(lrr, k=k, std_dev=0.2)


@pytest.mark.parametrize("f", [0.05, 0.2, 0.4])
@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("delta", [0.0, 0.3])
def test_emission_lrr_mcc_mean(f, k, delta):
    """The LRR emission is centred on the log2 copy ratio of the mixture, minus delta."""
    mu = np.log2(((1 - f) * k + 2 * f) / 2) - delta
    for lrr in [-1.0, 0.0, 0.4]:
        assert np.isclose(
            emission_lrr_mcc(lrr, k=k, f=f, delta=delta, std_dev=0.2),
            norm_logl(lrr, mu, 0.2),
        )
    assert emission_lrr_mcc(-9.0, k=k, f=f, delta=delta) == -1


def test_emission_lrr_mcc_nullisomy_floor():
    """With almost no maternal cells, nullisomy keeps the -3 floor of emission_lrr."""
    assert np.isclose(
        emission_lrr_mcc(-3.0, k=0, f=0.01, std_dev=0.2), norm_logl(-3.0, -3.0, 0.2)
    )
    # Enough maternal cells lift it above the floor.
    assert np.isclose(
        emission_lrr_mcc(-1.0, k=0, f=0.5, std_dev=0.2), norm_logl(-1.0, -1.0, 0.2)
    )


@pytest.mark.parametrize("s", [0.05, 0.15, 0.3, 1.2])
@pytest.mark.parametrize("x,m", [(0.0, 0.0), (0.3, -0.2), (-1.5, -1.0)])
def test_norm_logl_matches_scipy(x, m, s):
    """norm_logl is the normal log-density with standard deviation s."""
    assert np.isclose(norm_logl(x, m, s), norm.logpdf(x, loc=m, scale=s))


@pytest.mark.parametrize("f", [0.0, 0.2])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_emission_lrr_per_probe_sd(f, k):
    """Both LRR emissions use the per-probe standard deviation they are given."""
    rng = np.random.default_rng(1)
    sigmas = 0.16 * np.sqrt(rng.lognormal(sigma=1.0, size=50))
    lrrs = rng.normal(0.0, 0.3, size=50)
    mu = np.log2(((1 - f) * k + 2 * f) / 2)
    for lrr, s in zip(lrrs, sigmas):
        expected = norm.logpdf(lrr, loc=mu, scale=s)
        assert np.isclose(emission_lrr_mcc(lrr, k=k, f=f, std_dev=s), expected)
        if f == 0.0:
            assert np.isclose(emission_lrr(lrr, k=k, std_dev=s), expected)

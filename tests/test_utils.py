"""Testing module for utility functions."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from karyohmm_utils import (
    emission_baf,
    emission_depth,
    emission_reads,
    mat_dosage,
    pat_dosage,
)
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar
from scipy.stats import binom, poisson, truncnorm

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

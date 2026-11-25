"""Testing module for QuadHMM."""

import numpy as np
import pytest
from scipy.special import logsumexp

from karyohmm import PGTSim, QuadHMM

# --- Generating test data for applications --- #
pgt_sim = PGTSim()
data_disomy_sibs = pgt_sim.sibling_euploid_sim(
    m=4000, nsibs=3, std_dev=0.15, switch_err_rate=1e-2, seed=42
)
data_disomy_sibs_v2 = pgt_sim.sibling_euploid_sim(
    m=4000, nsibs=3, std_dev=0.2, switch_err_rate=1e-2, seed=24
)

data_disomy_sibs_v3 = pgt_sim.sibling_euploid_sim(
    m=4000, nsibs=3, std_dev=0.25, switch_err_rate=2e-2, seed=18
)


@pytest.mark.parametrize(
    "data", [data_disomy_sibs, data_disomy_sibs_v2, data_disomy_sibs_v3]
)
def test_forward_algorithm(data):
    """Test the forward algorithm implementation of the QuadHMM."""
    hmm = QuadHMM()
    _, _, _, _, loglik = hmm.forward_algorithm(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )


@pytest.mark.parametrize("data", [data_disomy_sibs])
def test_backward_algorithm(data):
    """Test the backward algorithm implementation of the QuadHMM."""
    hmm = QuadHMM()
    _, _, _, _, loglik = hmm.forward_algorithm(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )


@pytest.mark.parametrize("data", [data_disomy_sibs])
def test_forward_vs_backward_loglik(data):
    """Test that the log-likelihood from forward algorithm is equal to the backward."""
    hmm = QuadHMM()
    _, _, _, _, fwd_loglik = hmm.forward_algorithm(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    _, _, _, _, bwd_loglik = hmm.backward_algorithm(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    assert np.isclose(fwd_loglik, bwd_loglik, atol=1e-6)


@pytest.mark.parametrize(
    "data", [data_disomy_sibs, data_disomy_sibs_v2, data_disomy_sibs_v3]
)
def test_viterbi_algorithm(data):
    """Test the viterbi algorithm in the QuadHMM."""
    hmm = QuadHMM()
    path, states, deltas, psi = hmm.viterbi_algorithm(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    assert np.all((path >= 0) & (path < len(states)))
    res_path = hmm.restrict_path(path)
    assert np.all((res_path >= 0) & (res_path <= 3))


@pytest.mark.parametrize(
    "data", [data_disomy_sibs, data_disomy_sibs_v2, data_disomy_sibs_v3]
)
def test_recomb_isolation_map(data):
    """Test the viterbi algorithm in the QuadHMM."""
    hmm = QuadHMM()
    res_path01 = hmm.map_path(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    res_path02 = hmm.map_path(
        bafs=[data["baf_embryo0"], data["baf_embryo2"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    mat_rec, pat_rec, _, _ = hmm.isolate_recomb(res_path01, [res_path02])
    # True recombination events ...
    zs_maternal0 = data["zs_maternal0"]
    zs_paternal0 = data["zs_paternal0"]
    n_mat_rec = np.sum(zs_maternal0[:-1] != zs_maternal0[1:])
    n_pat_rec = np.sum(zs_paternal0[:-1] != zs_paternal0[1:])
    assert len(mat_rec) == n_mat_rec
    assert len(pat_rec) == n_pat_rec


@pytest.mark.parametrize(
    "data", [data_disomy_sibs, data_disomy_sibs_v2, data_disomy_sibs_v3]
)
def test_recomb_isolation_viterbi(data):
    """Test the viterbi algorithm in the QuadHMM."""
    hmm = QuadHMM()
    res_path01 = hmm.viterbi_path(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    res_path02 = hmm.viterbi_path(
        bafs=[data["baf_embryo0"], data["baf_embryo2"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    mat_rec, pat_rec, _, _ = hmm.isolate_recomb(res_path01, [res_path02])
    # True recombination events ...
    zs_maternal0 = data["zs_maternal0"]
    zs_paternal0 = data["zs_paternal0"]
    n_mat_rec = np.sum(zs_maternal0[:-1] != zs_maternal0[1:])
    n_pat_rec = np.sum(zs_paternal0[:-1] != zs_paternal0[1:])
    assert len(mat_rec) == n_mat_rec
    assert len(pat_rec) == n_pat_rec


@pytest.mark.parametrize(
    "data", [data_disomy_sibs, data_disomy_sibs_v2, data_disomy_sibs_v3]
)
def test_estimate_sharing(data):
    """Test the estimate of haplotype sharing across sibling embryos."""
    hmm = QuadHMM()
    mat_haplo_len, pat_haplo_len, both_haplo_len, tot_len = hmm.est_sharing(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    assert tot_len <= data["length"]
    assert tot_len == (data["pos"][-1] - data["pos"][0])


@pytest.mark.parametrize(
    "data", [data_disomy_sibs, data_disomy_sibs_v2, data_disomy_sibs_v3]
)
def test_flag_parental_genotype_errors(data):
    """Test that flag_parental_genotype_errors returns valid per-site error scores."""
    hmm = QuadHMM()
    gammas, states, _ = hmm.forward_backward(
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        pos=data["pos"],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    mat_err, pat_err = hmm.flag_parental_genotype_errors(
        gammas=gammas,
        states=states,
        bafs=[data["baf_embryo0"], data["baf_embryo1"]],
        mat_haps=data["mat_haps_true"],
        pat_haps=data["pat_haps_true"],
    )
    assert mat_err.shape == (data["pos"].size,)
    assert pat_err.shape == (data["pos"].size,)
    # Scores must be non-negative: the best alternative is always >= the called genotype
    assert np.all(mat_err >= 0.0)
    assert np.all(pat_err >= 0.0)
    assert np.all(np.isfinite(mat_err))
    assert np.all(np.isfinite(pat_err))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_viterbi_algorithm_traceback(seed):
    """The sibling Viterbi path follows its own backpointers and seeds site 0 with data.

    The traceback once read psi at column i rather than i + 1, delaying the path
    by a site, and site 0 was seeded with the prior alone. Driven by random BAFs
    and haplotypes because simulated sibling data never leaves its starting
    state, and an off-by-one traceback is invisible on a constant path.
    """
    hmm = QuadHMM()
    rng = np.random.default_rng(seed)
    n = 200
    kwargs = dict(
        bafs=[rng.uniform(size=n), rng.uniform(size=n)],
        pos=np.cumsum(rng.integers(1_000, 20_000, size=n)),
        mat_haps=rng.integers(0, 2, size=(2, n)),
        pat_haps=rng.integers(0, 2, size=(2, n)),
        r=1e-4,
    )
    path, _, deltas, psi = hmm.viterbi_algorithm(**kwargs)
    # The path switches, so the traceback is under test.
    assert np.unique(path).size > 1
    assert path[-1] == np.argmax(deltas[:, -1])
    for i in range(n - 1):
        assert path[i] == psi[path[i + 1], i + 1]
    # Site 0 carries the prior plus its emission, as in the forward pass.
    alphas, _, _, _, _ = hmm.forward_algorithm(**kwargs)
    assert np.allclose(deltas[:, 0] - logsumexp(deltas[:, 0]), alphas[:, 0])

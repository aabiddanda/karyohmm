"""Test suite for karyoHMM MetaHMM."""

import numpy as np
import pytest
from karyohmm_utils import (
    backward_algo_reads,
    backward_recursion,
    build_emission_matrix,
    build_emission_matrix_reads,
    create_index_arrays,
    forward_algo_reads,
    transition_kernel,
)
from scipy.special import logsumexp as logsumexp_sp

from karyohmm import MetaHMM, PGTSim

# --- Generating test data for applications --- #
pgt_sim = PGTSim()
data_disomy = pgt_sim.full_ploidy_sim(
    m=2000, ploidy=2, std_dev=0.15, mix_prop=0.7, seed=42
)
data_trisomy = pgt_sim.full_ploidy_sim(
    m=2000, ploidy=3, mat_skew=0, std_dev=0.15, mix_prop=0.7, seed=42
)
data_monosomy = pgt_sim.full_ploidy_sim(
    m=2000, ploidy=1, mat_skew=0, std_dev=0.15, mix_prop=0.7, seed=42
)
data_nullisomy = pgt_sim.full_ploidy_sim(
    m=2000, ploidy=0, mat_skew=0, std_dev=0.15, mix_prop=0.7, seed=42
)


def bph(states):
    """Identify states that are BPH - both parental homologs."""
    idx = []
    for i, s in enumerate(states):
        assert len(s) == 4
        k = 0
        for j in range(4):
            k += s[j] >= 0
        if k == 3:
            if s[1] != -1:
                if s[0] != s[1]:
                    # Both maternal homologs present
                    idx.append(i)
            if s[3] != -1:
                if s[2] != s[3]:
                    # Both paternal homologs present
                    idx.append(i)
    # Returns indices of both maternal & paternal BPH
    return idx


def sph(states):
    """Identify states that are SPH - single parental homolog."""
    idx = []
    for i, s in enumerate(states):
        assert len(s) == 4
        k = 0
        for j in range(4):
            k += s[j] >= 0
        if k == 3:
            if s[1] != -1:
                if s[0] == s[1]:
                    # Both maternal homologs present
                    idx.append(i)
            if s[3] != -1:
                if s[2] == s[3]:
                    # Both paternal homologs present
                    idx.append(i)
    # Returns indices of both maternal & paternal SPH
    return idx


def test_bph_sph():
    """Test that BPH vs. SPH give you the correct states."""
    hmm = MetaHMM()
    sph_idx = sph(hmm.states)
    bph_idx = bph(hmm.states)
    # The total number of states shouldb be 12
    assert len(sph_idx) == 8
    assert len(bph_idx) == 4
    for i in sph_idx:
        x = hmm.states[i]
        assert (x[0] == x[1]) or (x[2] == x[3])
    for i in bph_idx:
        x = hmm.states[i]
        assert (x[0] != x[1]) or (x[2] != x[3])


def test_data_integrity(data=data_disomy):
    """Test for some basic data sanity checks ..."""
    for x in ["baf", "mat_haps", "pat_haps"]:
        assert x in data
    baf = data["baf"]
    pos = data["pos"]
    mat_haps = data["mat_haps"]
    pat_haps = data["pat_haps"]
    assert np.all((baf <= 1.0) & (baf >= 0.0))
    assert baf.size == mat_haps.shape[1]
    assert baf.size == pos.size
    assert mat_haps.shape == pat_haps.shape
    assert np.all((mat_haps == 0) | (mat_haps == 1))
    assert np.all((pat_haps == 0) | (pat_haps == 1))


# --- Testing the metadata implementations --- #
@pytest.mark.parametrize("data", [data_disomy, data_trisomy, data_monosomy])
def test_forward_algorithm(data):
    """Test the implementation of the forward algorithm."""
    hmm = MetaHMM()
    _, _, _, karyotypes, loglik = hmm.forward_algorithm(
        bafs=data["baf"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
    )


@pytest.mark.parametrize("data", [data_disomy, data_trisomy, data_monosomy])
def test_backward_algorithm(data):
    """Test the implementation of the backward algorithm."""
    hmm = MetaHMM()
    _, _, _, karyotypes, loglik = hmm.backward_algorithm(
        bafs=data["baf"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
    )


@pytest.mark.parametrize("data", [data_disomy])
def test_disomy_model(data):
    """Test implementation under a pure-disomy model."""
    hmm = MetaHMM(disomy=True)
    _, _, _, karyotypes, loglik = hmm.forward_algorithm(
        bafs=data["baf"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
    )
    _, _, _, karyotypes, loglik = hmm.backward_algorithm(
        bafs=data["baf"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
    )
    gammas, _, karyotypes = hmm.forward_backward(
        bafs=data["baf"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
    )
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))


@pytest.mark.parametrize("data", [data_disomy])
def test_forward_vs_backward_loglik(data):
    """Test that the log-likelihood from forward algorithm is equal to the backward."""
    hmm = MetaHMM()

    _, _, _, _, fwd_loglik = hmm.forward_algorithm(
        bafs=data["baf"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
    )
    _, _, _, _, bwd_loglik = hmm.backward_algorithm(
        bafs=data["baf"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
    )
    assert np.isclose(fwd_loglik, bwd_loglik, atol=1e-6)


@pytest.mark.parametrize(
    "data", [data_disomy, data_trisomy, data_monosomy, data_nullisomy]
)
@pytest.mark.parametrize("mat_err,pat_err", [(0.0, 0.0), (1e-2, 2e-2)])
def test_forward_vs_backward_loglik_shared_emissions(data, mat_err, pat_err):
    """Forward and backward must agree across ploidies and both emission branches.

    They share build_emission_matrix, so a mismatch here means the two have
    drifted apart and any forward-backward posterior would be invalid.
    """
    hmm = MetaHMM()
    kwargs = dict(
        bafs=data["baf"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        mat_err=mat_err,
        pat_err=pat_err,
    )
    _, _, _, _, fwd_loglik = hmm.forward_algorithm(**kwargs)
    _, _, _, _, bwd_loglik = hmm.backward_algorithm(**kwargs)
    assert np.isclose(fwd_loglik, bwd_loglik, atol=1e-6)


@pytest.mark.parametrize("mat_err,pat_err", [(0.0, 0.0), (1e-2, 2e-2)])
def test_build_emission_matrix_shape_and_finiteness(mat_err, pat_err):
    """The shared emission matrix is (m x n) and finite on both branches."""
    hmm = MetaHMM()
    data = data_disomy
    emit = build_emission_matrix(
        data["baf"],
        data["lrr"],
        data["sigmas"],
        data["mat_haps"],
        data["pat_haps"],
        hmm.states,
        pi0=0.5,
        std_dev=0.25,
        mat_err=mat_err,
        pat_err=pat_err,
    )
    assert emit.shape == (len(hmm.states), data["baf"].size)
    assert np.all(np.isfinite(emit))


@pytest.mark.parametrize("ploidy", [0, 1, 2, 3])
def test_build_emission_matrix_reads_favors_true_ploidy(ploidy):
    """The read-level emission matrix is finite and peaks at the true ploidy.

    rec_rate is set low enough that the true path holds a single state across
    the chromosome, which is what makes summing a fixed state's emissions over
    all sites a valid comparison in the absence of the transition model.
    """
    hmm = MetaHMM()
    data = pgt_sim.full_ploidy_sim(
        m=1000, ploidy=ploidy, reads=True, coverage=10.0, rec_rate=1e-10, seed=42
    )
    emit = build_emission_matrix_reads(
        data["alt_reads"],
        data["ref_reads"],
        data["mat_haps"],
        data["pat_haps"],
        hmm.states,
        eps=1e-2,
        coverage=10.0,
        mismap_rate=1e-2,
    )
    assert emit.shape == (len(hmm.states), data["alt_reads"].size)
    assert np.all(np.isfinite(emit))
    ks = np.array([sum(s >= 0 for s in state) for state in hmm.states])
    assert ks[np.argmax(emit.sum(axis=1))] == ploidy


def test_build_emission_matrix_reads_masking():
    """Depth masking is per-channel and confined to the site it is applied to."""
    hmm = MetaHMM()
    data = pgt_sim.full_ploidy_sim(m=200, ploidy=2, reads=True, coverage=10.0, seed=42)
    alt, ref = data["alt_reads"], data["ref_reads"]
    kwargs = dict(
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        states=hmm.states,
        coverage=10.0,
    )
    depths = np.asarray(alt, dtype=int) + np.asarray(ref, dtype=int)
    emit = build_emission_matrix_reads(alt, ref, **kwargs)
    # Passing the implied depths explicitly must change nothing.
    assert np.array_equal(emit, build_emission_matrix_reads(alt, ref, depths=depths, **kwargs))
    depths[0] = -9
    masked = build_emission_matrix_reads(alt, ref, depths=depths, **kwargs)
    # -9 drops the depth term alone, so the site still discriminates states ...
    assert masked[:, 0].std() > 0.0
    assert np.array_equal(masked[:, 1:], emit[:, 1:])
    # ... and zeroing the reads as well drops the site entirely.
    alt0, ref0 = alt.copy(), ref.copy()
    alt0[0], ref0[0] = 0, 0
    full_mask = build_emission_matrix_reads(alt0, ref0, depths=depths, **kwargs)
    assert np.allclose(full_mask[:, 0], -1.0)


def ref_forward(emit, pos, karyotypes, r=1e-8, a=1e-2, unphased=0):
    """Forward recursion written unlike the engine: linear space, matrix product.

    The column maximum is factored out for stability. The engine holds the
    inter-site distance in a C float, so the distance is rounded to single
    precision here -- without that the two agree only to ~1e-8.
    """
    m, n = emit.shape
    K0, K1 = create_index_arrays(karyotypes)
    c = emit[:, 0].max()
    unnorm = np.exp(emit[:, 0] - c) / m
    scale = unnorm.sum()
    loglik = c + np.log(scale)
    alpha = unnorm / scale
    alphas = np.zeros((m, n))
    alphas[:, 0] = alpha
    for i in range(1, n):
        di = np.float32(pos[i] - pos[i - 1])
        A = np.exp(transition_kernel(K0, K1, d=di, r=r, a=a, unphased=unphased))
        c = emit[:, i].max()
        unnorm = np.exp(emit[:, i] - c) * (A.T @ alpha)
        scale = unnorm.sum()
        loglik += c + np.log(scale)
        alpha = unnorm / scale
        alphas[:, i] = alpha
    return alphas, loglik


@pytest.mark.parametrize("ploidy", [0, 1, 2, 3])
@pytest.mark.parametrize("disomy,unphased", [(False, 0), (False, 1), (True, 0)])
def test_forward_algo_reads_vs_reference(ploidy, disomy, unphased):
    """The read-based forward pass matches an independently written recursion."""
    hmm = MetaHMM(disomy=disomy)
    data = pgt_sim.full_ploidy_sim(m=300, ploidy=ploidy, reads=True, coverage=10.0, seed=42)
    kwargs = dict(eps=1e-2, coverage=10.0, mismap_rate=1e-2)
    alphas, scaler, _, _, loglik = forward_algo_reads(
        data["alt_reads"],
        data["ref_reads"],
        data["pos"],
        data["mat_haps"],
        data["pat_haps"],
        hmm.states,
        hmm.karyotypes,
        unphased=unphased,
        **kwargs,
    )
    emit = build_emission_matrix_reads(
        data["alt_reads"],
        data["ref_reads"],
        data["mat_haps"],
        data["pat_haps"],
        hmm.states,
        **kwargs,
    )
    ref_alphas, ref_loglik = ref_forward(
        emit, data["pos"], hmm.karyotypes, unphased=unphased
    )
    assert np.isclose(loglik, sum(scaler))
    assert np.all(np.isclose(np.sum(np.exp(alphas), axis=0), 1.0))
    assert np.isclose(loglik, ref_loglik, atol=1e-6)
    assert np.allclose(np.exp(alphas), ref_alphas, atol=1e-10)


def ref_backward(emit, pos, karyotypes, r=1e-8, a=1e-2, unphased=0):
    """Backward recursion written unlike the engine: linear space, matrix product.

    The transition is applied as ``A @ v``: beta_i(j) sums over where j can go,
    so it needs row j of the kernel.  That distinction is not cosmetic here --
    transition_kernel is asymmetric, because create_index_arrays normalises the
    inter-karyotype entry by the *source* class size, K1[i, j] = 1/(m - ks[i]),
    making the 0 -> 3m rate 1/20 against 1/15 the other way.
    """
    m, n = emit.shape
    K0, K1 = create_index_arrays(karyotypes)
    beta = np.ones(m)
    logscale = 0.0
    betas = np.zeros((m, n))
    betas[:, -1] = beta / beta.sum()
    for i in range(n - 2, -1, -1):
        di = np.float32(pos[i + 1] - pos[i])
        A = np.exp(transition_kernel(K0, K1, d=di, r=r, a=a, unphased=unphased))
        c = emit[:, i + 1].max()
        beta = A @ (np.exp(emit[:, i + 1] - c) * beta)
        scale = beta.sum()
        beta /= scale
        logscale += c + np.log(scale)
        betas[:, i] = beta
    c = emit[:, 0].max()
    loglik = logscale - np.log(m) + c + np.log((np.exp(emit[:, 0] - c) * beta).sum())
    return betas, loglik


@pytest.mark.parametrize("upd", [False, True])
def test_backward_recursion_conserves_mass(upd):
    """With nothing to learn from the data, the backward pass must stay uniform.

    Under a flat emission matrix beta_i(j) = sum_k A[j,k] * const = const,
    because each row of the kernel sums to 1 exactly.  Indexing the column
    instead breaks this: transition_kernel is asymmetric, so its columns sum to
    1 only to ~5e-7 and the recursion leaks mass at every site.  This pins the
    row-vs-column choice without reference to any data.
    """
    hmm = MetaHMM(upd=upd)
    pos = np.arange(1, 401) * 25000.0
    flat = np.zeros((len(hmm.states), pos.size))
    betas, _ = backward_recursion(flat, pos, hmm.karyotypes)
    # Site 0 additionally carries the prior and its own emission.
    assert np.allclose(np.exp(betas[:, :-1]), 1.0 / len(hmm.states), atol=1e-12)


@pytest.mark.parametrize("ploidy", [0, 1, 2, 3])
@pytest.mark.parametrize("disomy,unphased", [(False, 0), (False, 1), (True, 0)])
def test_backward_algo_reads_vs_reference(ploidy, disomy, unphased):
    """The read-based backward pass matches an independently written recursion."""
    hmm = MetaHMM(disomy=disomy)
    data = pgt_sim.full_ploidy_sim(m=300, ploidy=ploidy, reads=True, coverage=10.0, seed=42)
    kwargs = dict(eps=1e-2, coverage=10.0, mismap_rate=1e-2)
    betas, scaler, _, _, loglik = backward_algo_reads(
        data["alt_reads"],
        data["ref_reads"],
        data["pos"],
        data["mat_haps"],
        data["pat_haps"],
        hmm.states,
        hmm.karyotypes,
        unphased=unphased,
        **kwargs,
    )
    emit = build_emission_matrix_reads(
        data["alt_reads"],
        data["ref_reads"],
        data["mat_haps"],
        data["pat_haps"],
        hmm.states,
        **kwargs,
    )
    ref_betas, ref_loglik = ref_backward(
        emit, data["pos"], hmm.karyotypes, unphased=unphased
    )
    assert np.isclose(loglik, sum(scaler))
    assert np.isclose(loglik, ref_loglik, atol=1e-6)
    # Site 0 carries the prior and its own emission, so only 1: is a pure beta.
    assert np.allclose(np.exp(betas[:, 1:]), ref_betas[:, 1:], atol=1e-10)


@pytest.mark.parametrize("ploidy", [0, 1, 2, 3])
@pytest.mark.parametrize("unphased", [0, 1])
def test_forward_vs_backward_loglik_reads(ploidy, unphased):
    """The read-based forward and backward passes must agree on the loglik.

    They share build_emission_matrix_reads, so a mismatch means the two have
    drifted apart and any forward-backward posterior would be invalid.
    """
    hmm = MetaHMM()
    data = pgt_sim.full_ploidy_sim(m=500, ploidy=ploidy, reads=True, coverage=10.0, seed=42)
    kwargs = dict(
        alt_reads=data["alt_reads"],
        ref_reads=data["ref_reads"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        eps=1e-2,
        coverage=10.0,
        mismap_rate=1e-2,
        unphased=bool(unphased),
    )
    _, _, _, _, fwd_loglik = hmm.forward_algorithm_reads(**kwargs)
    _, _, _, _, bwd_loglik = hmm.backward_algorithm_reads(**kwargs)
    assert np.isclose(fwd_loglik, bwd_loglik, atol=1e-6)


@pytest.mark.parametrize("ploidy", [0, 1, 2, 3])
def test_forward_backward_reads_ploidy_correctness(ploidy):
    """Combining the two read-based passes gives a valid posterior on the truth.

    See test_forward_algorithm_reads_ploidy_correctness on why rec_rate is set
    to the model's own r here.
    """
    hmm = MetaHMM()
    data = pgt_sim.full_ploidy_sim(
        m=1000, ploidy=ploidy, reads=True, coverage=10.0, rec_rate=1e-8, seed=42
    )
    kwargs = dict(
        alt_reads=data["alt_reads"],
        ref_reads=data["ref_reads"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        eps=1e-2,
        coverage=10.0,
        mismap_rate=1e-2,
    )
    alphas, _, _, karyotypes, _ = hmm.forward_algorithm_reads(**kwargs)
    betas, _, _, _, _ = hmm.backward_algorithm_reads(**kwargs)
    gammas = (alphas + betas) - logsumexp_sp(alphas + betas, axis=0)
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))
    post = hmm.posterior_karyotypes(gammas, karyotypes)
    assert max(post, key=post.get) == data["aploid"]
    assert post[data["aploid"]] > 0.95


@pytest.mark.parametrize("ploidy", [0, 1, 2, 3])
def test_forward_algorithm_reads_ploidy_correctness(ploidy):
    """Read-level inference recovers the simulated karyotype.

    Forward-only, so the statistic is the filtered posterior averaged over
    sites -- the same time-average posterior_karyotypes takes over gammas.

    rec_rate is set to the model's own r rather than the simulator default of
    1e-4, which over ~10kb site spacing switches the transmitted haplotype at
    nearly every site.  The read emission is far sharper than the BAF one at
    this coverage, so it pays a correspondingly larger penalty for a haplotype
    the frozen disomy states cannot follow, and buys the fit back with a BPH
    trisomy state that carries both homologs.
    """
    hmm = MetaHMM()
    data = pgt_sim.full_ploidy_sim(
        m=1000, ploidy=ploidy, reads=True, coverage=10.0, rec_rate=1e-8, seed=42
    )
    alphas, _, _, karyotypes, _ = hmm.forward_algorithm_reads(
        alt_reads=data["alt_reads"],
        ref_reads=data["ref_reads"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        eps=1e-2,
        coverage=10.0,
        mismap_rate=1e-2,
    )
    filt = np.exp(alphas)
    assert np.all(np.isclose(filt.sum(axis=0), 1.0))
    post = {
        k: filt[karyotypes == k, :].sum() / filt.shape[1]
        for k in np.unique(karyotypes)
    }
    assert max(post, key=post.get) == data["aploid"]
    assert post[data["aploid"]] > 0.95


@pytest.mark.parametrize("data", [data_disomy, data_trisomy, data_monosomy])
def test_fwd_bwd_algorithm(data):
    """Test the properties of the output from the fwd-bwd algorithm."""
    hmm = MetaHMM()
    gammas, _, karyotypes = hmm.forward_backward(
        bafs=data["baf"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
    )
    # all of the columns must have a sum to 1
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))
    post_dict = hmm.posterior_karyotypes(gammas, karyotypes)
    for x in ["0", "1m", "1p", "2", "3m", "3p"]:
        assert x in post_dict
    assert np.isclose(sum([post_dict[k] for k in post_dict]), 1.0)


@pytest.mark.parametrize("data", [data_disomy, data_trisomy, data_monosomy])
def test_est_pi0_sigma(data):
    """Test the optimization routine on the forward-algorithm likelihood."""
    hmm = MetaHMM()
    pi0_est, sigma_est = hmm.est_sigma_pi0(
        bafs=data["baf"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
    )
    assert (pi0_est > 0) and (pi0_est < 1.0)
    assert (sigma_est > 0) and (sigma_est < 1.0)


@pytest.mark.parametrize(
    "data,pi0_bounds",
    [
        (data_disomy, (0, 1)),
        (data_disomy, 0.1),
        (data_disomy, (0.1, 0.5, 0.9)),
        (data_disomy, (0.99, 0.01)),
    ],
)
def test_est_pi0_sigma_bad_pi0_bounds(data, pi0_bounds):
    """Test the pi0 bounds as input to the MLE estimation."""
    with pytest.raises(Exception):
        hmm = MetaHMM()
        pi0_est, sigma_est = hmm.est_sigma_pi0(
            bafs=data["baf"],
            pos=data["pos"],
            mat_haps=data["mat_haps"],
            pat_haps=data["pat_haps"],
            lrrs=data["lrr"],
            sigmas=data["sigmas"],
            pi0_bounds=pi0_bounds,
        )


@pytest.mark.parametrize(
    "data,sigma_bounds",
    [(data_disomy, (0, 1)), (data_disomy, 0.1), (data_disomy, (0.99, 0.01))],
)
def test_est_pi0_sigma_bad_sigma_bounds(data, sigma_bounds):
    """Test the sigma bounds as input to the MLE estimation."""
    with pytest.raises(Exception):
        hmm = MetaHMM()
        pi0_est, sigma_est = hmm.est_sigma_pi0(
            bafs=data["baf"],
            pos=data["pos"],
            mat_haps=data["mat_haps"],
            pat_haps=data["pat_haps"],
            lrrs=data["lrr"],
            sigmas=data["sigmas"],
            sigma_bounds=sigma_bounds,
        )


@pytest.mark.parametrize(
    "data,algo",
    [(data_disomy, "Powell"), (data_disomy, "L-BFGS-B"), (data_disomy, "Nelder-Mead")],
)
def test_est_pi0_sigma_algos(data, algo):
    """Test optimization of parameter estimates using different algorithms."""
    hmm = MetaHMM()
    pi0_est, sigma_est = hmm.est_sigma_pi0(
        bafs=data["baf"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
        algo=algo,
    )
    assert (pi0_est > 0) and (pi0_est < 1.0)
    assert (sigma_est > 0) and (sigma_est < 1.0)


def test_string_rep():
    """Test that the string representation of states makes sense."""
    hmm = MetaHMM()
    for s in hmm.states:
        x = hmm.get_state_str(s)
        m = sum([j >= 0 for j in s])
        if m == 0:
            assert x == "0"
        elif m == 1:
            if x[0] == "m":
                assert s in hmm.m_monosomy_states
            else:
                assert s in hmm.p_monosomy_states
        elif m == 2:
            assert len(x) == 2 * m
            assert x in ["m0p0", "m0p1", "m1p0", "m1p1"]
        else:
            assert len(x) == 2 * m
            assert (
                ("m0m1" in x)
                | ("p0p1" in x)
                | ("m0m0" in x)
                | ("m1m1" in x)
                | ("p0p0" in x)
                | ("p1p1" in x)
            )


@pytest.mark.parametrize("r,a", [(1e-8, 1e-2), (1e-8, 1e-4), (1e-8, 1e-6)])
def test_transition_matrices(r, a):
    """Test that transition matrices obey the rules."""
    hmm = MetaHMM()
    K0, K1 = create_index_arrays(hmm.karyotypes)
    A = transition_kernel(K0, K1, r=r, a=a)
    for i in range(A.shape[0]):
        assert np.isclose(np.exp(logsumexp_sp(A[i, :])), 1.0)


@pytest.mark.parametrize(
    "r,a,d",
    [(1e-8, 1e-2, 1e5), (1e-8, 1e-3, 1e4), (1e-8, 1e-2, 1e3), (1e-8, 1e-1, 1e8)],
)
def test_transition_matrices_dist(r, a, d):
    """Test how transition matrices scale with distance."""
    assert r < a
    hmm = MetaHMM()
    K0, K1 = create_index_arrays(hmm.karyotypes)
    A = transition_kernel(K0, K1, d=d, r=r, a=a)
    for i in range(A.shape[0]):
        assert np.isclose(np.exp(logsumexp_sp(A[i, :])), 1.0)


@pytest.mark.parametrize(
    "data", [data_disomy, data_trisomy, data_monosomy, data_nullisomy]
)
def test_ploidy_correctness(data):
    """This actually tests that the posterior inference of whole-chromosome aneuploidies is correct here."""
    hmm = MetaHMM()
    gammas, _, karyotypes = hmm.forward_backward(
        bafs=data["baf"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
    )
    # all of the columns must have a sum to 1
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))
    post_dict = hmm.posterior_karyotypes(gammas, karyotypes)
    max_post = np.max([post_dict[p] for p in post_dict])
    for x in ["0", "1m", "1p", "2", "3m", "3p"]:
        assert x in post_dict
    assert post_dict[data["aploid"]] == max_post
    assert post_dict[data["aploid"]] > 0.95


@pytest.mark.parametrize(
    "data", [data_disomy, data_trisomy, data_monosomy, data_nullisomy]
)
def test_ploidy_correctness_mle(data):
    """This actually tests that the posterior inference of whole-chromosome aneuploidies is correct under mle."""
    hmm = MetaHMM()
    pi0_est, sigma_est = hmm.est_sigma_pi0(
        bafs=data["baf"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
        algo="Powell",
    )
    gammas, _, karyotypes = hmm.forward_backward(
        bafs=data["baf"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        pi0=pi0_est,
        std_dev=sigma_est,
    )
    # all of the columns must have a sum to 1
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))
    post_dict = hmm.posterior_karyotypes(gammas, karyotypes)
    max_post = np.max([post_dict[p] for p in post_dict])
    for x in ["0", "1m", "1p", "2", "3m", "3p"]:
        assert x in post_dict
    assert post_dict[data["aploid"]] == max_post
    assert post_dict[data["aploid"]] > 0.95


@pytest.mark.parametrize("data", [data_disomy])
def test_embryo_genotype(data):
    """Test that the embryo genotyping works as intended."""
    hmm = MetaHMM(disomy=True)
    pi0_est, sigma_est = hmm.est_sigma_pi0(
        bafs=data["baf"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
        algo="Powell",
    )
    dosages = hmm.genotype_embryo(
        bafs=data["baf"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
        pi0=pi0_est,
        std_dev=sigma_est,
    )
    # Check that all dosages sum to 1 ...
    for i in range(dosages.shape[1]):
        assert np.isclose(dosages[:, i].sum(), 1.0)


@pytest.mark.parametrize("data", [data_disomy, data_trisomy, data_monosomy])
def test_flag_parental_genotype_errors(data):
    """Test that flag_parental_genotype_errors returns valid per-site error scores."""
    hmm = MetaHMM()
    gammas, states, _ = hmm.forward_backward(
        bafs=data["baf"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
        pos=data["pos"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
    )
    mat_err, pat_err = hmm.flag_parental_genotype_errors(
        gammas=gammas,
        states=states,
        bafs=data["baf"],
        mat_haps=data["mat_haps"],
        pat_haps=data["pat_haps"],
    )
    assert mat_err.shape == (data["baf"].size,)
    assert pat_err.shape == (data["baf"].size,)
    # Scores must be non-negative: the best alternative is always >= the called genotype
    assert np.all(mat_err >= 0.0)
    assert np.all(pat_err >= 0.0)
    # Homozygous parental sites cannot benefit from flipping the other allele when the
    # same allele is already the best option, so scores should be finite
    assert np.all(np.isfinite(mat_err))
    assert np.all(np.isfinite(pat_err))

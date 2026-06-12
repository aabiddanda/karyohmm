"""Test suite for karyoHMM PocHMM."""

from karyohmm import PGTSim, PocHMM
import numpy as np
import pytest


# --- Generating test data for applications in the PocHMM (duo) setting --- #
pgt_sim = PGTSim()
data_disomy = pgt_sim.full_ploidy_sim(m=1000, mix_prop=0.7, std_dev=0.1, seed=42)
data_nullisomy = pgt_sim.full_ploidy_sim(
    m=1000, ploidy=0, mat_skew=1.0, mix_prop=0.7, std_dev=0.1, seed=42
)
# Trisomy: maternal-origin (3m) and paternal-origin (3p)
data_mat_trisomy = pgt_sim.full_ploidy_sim(
    m=1000, ploidy=3, mat_skew=1.0, mix_prop=0.7, std_dev=0.1, seed=42
)
data_pat_trisomy = pgt_sim.full_ploidy_sim(
    m=1000, ploidy=3, mat_skew=0.0, mix_prop=0.7, std_dev=0.1, seed=42
)
# Monosomy: maternal-origin (1p: maternal chrom lost, paternal survives) and
#           paternal-origin (1m: paternal chrom lost, maternal survives)
data_mat_origin_mono = pgt_sim.full_ploidy_sim(
    m=1000, ploidy=1, mat_skew=1.0, mix_prop=0.7, std_dev=0.1, seed=42
)
data_pat_origin_mono = pgt_sim.full_ploidy_sim(
    m=1000, ploidy=1, mat_skew=0.0, mix_prop=0.7, std_dev=0.1, seed=42
)

# Convenience alias kept for parametrize tests that cover all karyotypes
_all_data = [
    data_disomy,
    data_nullisomy,
    data_mat_trisomy,
    data_pat_trisomy,
    data_mat_origin_mono,
    data_pat_origin_mono,
]


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
                    idx.append(i)
            if s[3] != -1:
                if s[2] != s[3]:
                    idx.append(i)
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
                    idx.append(i)
            if s[3] != -1:
                if s[2] == s[3]:
                    idx.append(i)
    return idx


def test_bph_sph():
    """Test that BPH vs. SPH give you the correct states."""
    hmm = PocHMM()
    sph_idx = sph(hmm.states)
    bph_idx = bph(hmm.states)
    assert len(sph_idx) == 8
    assert len(bph_idx) == 4
    for i in sph_idx:
        x = hmm.states[i]
        assert (x[0] == x[1]) or (x[2] == x[3])
    for i in bph_idx:
        x = hmm.states[i]
        assert (x[0] != x[1]) or (x[2] != x[3])


@pytest.mark.parametrize("data", _all_data)
def test_data_integrity(data):
    """Test basic data sanity checks."""
    for x in ["baf", "lrr", "sigmas", "mat_haps", "pat_haps"]:
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


@pytest.mark.parametrize("data", _all_data)
def test_pochmm_forward(data):
    """Forward loglik changes when LRR is included vs. blanked."""
    baf = data["baf"]
    lrr = data["lrr"]
    sigmas = data["sigmas"]
    pos = data["pos"]
    mat_haps = data["mat_haps"]
    pochmm = PocHMM()
    _, _, _, _, loglik = pochmm.forward_algorithm(
        bafs=baf, lrrs=lrr, sigmas=sigmas, pos=pos, haps=mat_haps, freqs=None
    )
    _, _, _, _, loglik_no_lrr = pochmm.forward_algorithm(
        bafs=baf,
        lrrs=np.repeat(-9, baf.size),
        sigmas=np.ones(baf.size),
        pos=pos,
        haps=mat_haps,
        freqs=None,
    )
    assert loglik != loglik_no_lrr


@pytest.mark.parametrize("data", _all_data)
def test_pochmm_backward(data):
    """Backward loglik changes when LRR is included vs. blanked."""
    baf = data["baf"]
    lrr = data["lrr"]
    sigmas = data["sigmas"]
    pos = data["pos"]
    mat_haps = data["mat_haps"]
    pochmm = PocHMM()
    _, _, _, _, loglik = pochmm.backward_algorithm(
        bafs=baf, lrrs=lrr, sigmas=sigmas, pos=pos, haps=mat_haps, freqs=None
    )
    _, _, _, _, loglik_no_lrr = pochmm.backward_algorithm(
        bafs=baf,
        lrrs=np.repeat(-9, baf.size),
        sigmas=np.ones(baf.size),
        pos=pos,
        haps=mat_haps,
        freqs=None,
    )
    assert loglik != loglik_no_lrr


@pytest.mark.parametrize("data", _all_data)
def test_pochmm_fwd_bwd(data):
    """Forward and backward log-likelihoods agree."""
    baf = data["baf"]
    lrr = data["lrr"]
    sigmas = data["sigmas"]
    pos = data["pos"]
    mat_haps = data["mat_haps"]
    pochmm = PocHMM()
    _, _, _, _, loglik = pochmm.forward_algorithm(
        bafs=baf, lrrs=lrr, sigmas=sigmas, pos=pos, haps=mat_haps, freqs=None
    )
    _, _, _, _, loglik2 = pochmm.backward_algorithm(
        bafs=baf, lrrs=lrr, sigmas=sigmas, pos=pos, haps=mat_haps, freqs=None
    )
    assert np.isclose(loglik, loglik2)


@pytest.mark.parametrize("data", _all_data)
def test_pochmm_fwdbwd(data):
    """Forward-backward posteriors are valid probability distributions."""
    baf = data["baf"]
    lrr = data["lrr"]
    sigmas = data["sigmas"]
    pos = data["pos"]
    mat_haps = data["mat_haps"]
    pochmm = PocHMM()
    gammas, _, _ = pochmm.forward_backward(
        bafs=baf, lrrs=lrr, sigmas=sigmas, pos=pos, haps=mat_haps, freqs=None
    )
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))
    pp = np.exp(gammas)
    assert np.all((pp >= 0.0) & (pp <= 1.0))


@pytest.mark.parametrize("use_lrr", [True, False], ids=["with_lrr", "no_lrr"])
@pytest.mark.parametrize(
    "data",
    [
        data_disomy,
        data_mat_trisomy,
        data_pat_trisomy,
        data_mat_origin_mono,
        data_pat_origin_mono,
    ],
    ids=["disomy", "mat_trisomy", "pat_trisomy", "mat_origin_mono", "pat_origin_mono"],
)
def test_pochmm_ploidy_correctness(data, use_lrr):
    """Argmax karyotype matches the true simulated karyotype, with and without LRR."""
    baf = data["baf"]
    lrr = data["lrr"] if use_lrr else np.repeat(-9.0, baf.size)
    sigmas = data["sigmas"] if use_lrr else np.ones(baf.size)
    pos = data["pos"]
    mat_haps = data["mat_haps"]
    # Use paternal allele frequencies as a proxy for an external reference panel
    freqs = data["pat_haps"].sum(axis=0) / 2.0
    hmm = PocHMM()
    gammas, _, karyotypes = hmm.forward_backward(
        bafs=baf,
        lrrs=lrr,
        sigmas=sigmas,
        pos=pos,
        haps=mat_haps,
        pi0=0.7,
        std_dev=0.1,
        freqs=freqs,
    )
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))
    post_dict = hmm.posterior_karyotypes(gammas, karyotypes)
    for x in ["0", "1m", "1p", "2", "3m", "3p"]:
        assert x in post_dict
    max_post = max(post_dict.values())
    assert post_dict[data["aploid"]] == max_post

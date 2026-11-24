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

# --- Reference-panel test data ---
# Simulate a reference panel of 200 haplotypes (100 diploid samples), then draw
# maternal and paternal haplotypes directly from that panel to simulate an embryo.
# This mirrors realistic usage of sim_haplotype_ref_panel + infer_missing_af.
np.random.seed(7)
_base_haps, _, _ = pgt_sim.draw_parental_genotypes(m=1000, seed=7)
_base_pos = np.sort(np.random.uniform(high=1e7, size=1000))
_ref_panel_200 = pgt_sim.sim_haplotype_ref_panel(
    _base_haps, _base_pos, panel_size=200, seed=7
)
# Rows 0-1 → maternal haplotypes; rows 2-3 → paternal haplotypes
_panel_mat = _ref_panel_200[:2, :].astype(np.uint16)
_panel_pat = _ref_panel_200[2:4, :].astype(np.uint16)
_, _, _panel_mat_hap1, _panel_pat_hap1, _panel_aploid = pgt_sim.sim_haplotype_paths(
    _panel_mat, _panel_pat, _base_pos, ploidy=2, seed=7
)
_, _panel_baf = pgt_sim.sim_b_allele_freq(
    _panel_mat_hap1, _panel_pat_hap1, ploidy=2, std_dev=0.1, mix_prop=0.7, seed=7
)
_panel_lrr, _panel_sigmas = pgt_sim.sim_logR_ratio(
    _panel_mat_hap1, _panel_pat_hap1, ploidy=2, seed=7
)
data_from_ref_panel = {
    "mat_haps": _panel_mat,
    "pat_haps": _panel_pat,
    "baf": _panel_baf,
    "lrr": _panel_lrr,
    "sigmas": _panel_sigmas,
    "pos": _base_pos,
    "aploid": _panel_aploid,
    "ref_panel": _ref_panel_200,
}

# Map karyotype labels to integer ploidy levels (for tests that check ploidy
# without requiring exact parental-origin identification).
_PLOIDY_LEVEL = {"0": 0, "1m": 1, "1p": 1, "2": 2, "3m": 3, "3p": 3}


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
def test_pochmm_ploidy_correctness_uniform_freqs(data, use_lrr):
    """Ploidy level is correctly identified using a uniform (even) prior on allele frequencies.

    With freqs=0.5 at all sites the model lacks information to distinguish parental
    origin, so only the integer ploidy level (0/1/2/3) is asserted.
    """
    baf = data["baf"]
    lrr = data["lrr"] if use_lrr else np.repeat(-9.0, baf.size)
    sigmas = data["sigmas"] if use_lrr else np.ones(baf.size)
    freqs = np.full(baf.size, 0.5)
    hmm = PocHMM()
    gammas, _, karyotypes = hmm.forward_backward(
        bafs=baf,
        lrrs=lrr,
        sigmas=sigmas,
        pos=data["pos"],
        haps=data["mat_haps"],
        pi0=0.7,
        std_dev=0.1,
        freqs=freqs,
    )
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))
    post_dict = hmm.posterior_karyotypes(gammas, karyotypes)
    for x in ["0", "1m", "1p", "2", "3m", "3p"]:
        assert x in post_dict
    best = max(post_dict, key=post_dict.get)
    assert _PLOIDY_LEVEL[str(best)] == _PLOIDY_LEVEL[str(data["aploid"])]


def test_pochmm_ploidy_correctness_from_ref_panel():
    """Exact karyotype is recovered when parents are drawn from a haplotype reference panel
    and paternal allele frequencies are inferred via infer_missing_af instead of using
    the true paternal haplotypes.

    Setup: 200 haplotypes (100 diploid samples) are simulated with
    sim_haplotype_ref_panel; rows 0-1 become the maternal haplotypes and rows 2-3
    the paternal haplotypes for the embryo simulation.
    """
    data = data_from_ref_panel
    hmm = PocHMM()
    mat_geno = data["mat_haps"].sum(axis=0).astype(np.int32)
    freqs = hmm.infer_missing_af(data["baf"], mat_geno, data["ref_panel"], data["pos"])
    assert freqs.shape == data["baf"].shape
    assert np.all((freqs >= 0.0) & (freqs <= 1.0))
    gammas, _, karyotypes = hmm.forward_backward(
        bafs=data["baf"],
        lrrs=data["lrr"],
        sigmas=data["sigmas"],
        pos=data["pos"],
        haps=data["mat_haps"],
        pi0=0.7,
        std_dev=0.1,
        freqs=freqs,
    )
    assert np.all(np.isclose(np.sum(np.exp(gammas), axis=0), 1.0))
    post_dict = hmm.posterior_karyotypes(gammas, karyotypes)
    for x in ["0", "1m", "1p", "2", "3m", "3p"]:
        assert x in post_dict
    assert post_dict[data["aploid"]] == max(post_dict.values())


def test_infer_missing_af_properties():
    """infer_missing_af returns a valid allele-frequency array of the correct shape."""
    baf = data_disomy["baf"]
    mat_geno = data_disomy["mat_haps"].sum(axis=0).astype(np.int32)
    panel = pgt_sim.sim_haplotype_ref_panel(
        data_disomy["mat_haps"], data_disomy["pos"], panel_size=50, seed=1
    )
    hmm = PocHMM()
    freqs = hmm.infer_missing_af(baf, mat_geno, panel, data_disomy["pos"])
    assert freqs.shape == baf.shape
    assert np.all((freqs >= 0.0) & (freqs <= 1.0))
    assert not np.any(np.isnan(freqs))


def test_infer_missing_af_no_anchors_fallback():
    """infer_missing_af falls back to the panel mean when no anchor sites exist."""
    baf = data_disomy["baf"]
    panel = pgt_sim.sim_haplotype_ref_panel(
        data_disomy["mat_haps"], data_disomy["pos"], panel_size=50, seed=1
    )
    hmm = PocHMM()
    # Force no anchor sites by making all genotypes heterozygous (geno==1 never triggers)
    het_geno = np.ones(baf.size, dtype=np.int32)
    freqs = hmm.infer_missing_af(baf, het_geno, panel, data_disomy["pos"])
    expected = panel.mean(axis=0)
    assert freqs.shape == baf.shape
    assert np.allclose(freqs, expected)


def test_genotype_parent():
    """genotype_parent returns a (3, m) log-posterior matrix that sums to 1 per site."""
    baf = data_disomy["baf"]
    lrr = data_disomy["lrr"]
    sigmas = data_disomy["sigmas"]
    hmm = PocHMM()
    gammas, _, _ = hmm.forward_backward(
        bafs=baf,
        lrrs=lrr,
        sigmas=sigmas,
        pos=data_disomy["pos"],
        haps=data_disomy["mat_haps"],
        pi0=0.7,
        std_dev=0.1,
    )
    geno_dosage = hmm.genotype_parent(
        bafs=baf,
        lrrs=lrr,
        sigmas=sigmas,
        haps=data_disomy["mat_haps"],
        gammas=gammas,
        pi0=0.7,
        std_dev=0.1,
    )
    assert geno_dosage.shape == (3, baf.size)
    col_sums = np.exp(geno_dosage).sum(axis=0)
    assert np.allclose(col_sums, 1.0, atol=1e-4)
    assert np.all((np.exp(geno_dosage) >= 0.0) & (np.exp(geno_dosage) <= 1.0 + 1e-10))


# --- Maternal cell contamination (mcc) and the LRR offset (delta) --- #
data_mcc_1p = pgt_sim.full_ploidy_sim(
    m=2000,
    ploidy=1,
    mat_skew=1.0,
    length=2e7,
    rec_rate=1e-8,
    std_dev=0.12,
    mcc=0.3,
    seed=1,
)


@pytest.mark.parametrize("maternal", [True, False])
@pytest.mark.parametrize("obs_err", [0.0, 1e-2])
@pytest.mark.parametrize("mcc,delta", [(0.15, 0.0), (0.3, 0.2)])
def test_pochmm_fwd_bwd_mcc(maternal, obs_err, mcc, delta):
    """Forward and backward log-likelihoods agree under contamination and an LRR offset."""
    data = data_mat_trisomy
    kwargs = {
        "bafs": data["baf"],
        "lrrs": data["lrr"],
        "sigmas": data["sigmas"],
        "pos": data["pos"],
        "haps": data["mat_haps"] if maternal else data["pat_haps"],
        "freqs": data["af"],
        "maternal": maternal,
        "obs_err": obs_err,
        "mcc": mcc,
        "delta": delta,
    }
    pochmm = PocHMM()
    _, _, _, _, loglik = pochmm.forward_algorithm(**kwargs)
    _, _, _, _, loglik2 = pochmm.backward_algorithm(**kwargs)
    assert np.isclose(loglik, loglik2)
    gammas, _, _ = pochmm.forward_backward(**kwargs)
    assert np.allclose(np.exp(gammas).sum(axis=0), 1.0)


@pytest.mark.parametrize("shift", [-0.3, 0.25])
def test_pochmm_lrr_offset(shift):
    """Shifting every LRR by -delta and setting the offset to delta leaves the likelihood unchanged."""
    data = data_mat_trisomy
    kwargs = {
        "bafs": data["baf"],
        "sigmas": data["sigmas"],
        "pos": data["pos"],
        "haps": data["mat_haps"],
        "freqs": data["af"],
        "mcc": 0.1,
    }
    pochmm = PocHMM()
    *_, base = pochmm.forward_algorithm(lrrs=data["lrr"], delta=0.0, **kwargs)
    *_, offset = pochmm.forward_algorithm(
        lrrs=data["lrr"] - shift, delta=shift, **kwargs
    )
    *_, unmatched = pochmm.forward_algorithm(
        lrrs=data["lrr"] - shift, delta=0.0, **kwargs
    )
    assert np.isclose(base, offset)
    assert unmatched < base


def test_pochmm_mcc_rescues_paternal_monosomy():
    """With BAF alone, 30% MCC makes 1p look like 3p unless the model accounts for it."""
    data = data_mcc_1p
    m = data["baf"].size
    kwargs = {
        "bafs": data["baf"],
        "lrrs": np.full(m, -9.0),
        "sigmas": np.ones(m),
        "pos": data["pos"],
        "haps": data["mat_haps"],
        "freqs": data["af"],
        "pi0": 0.3,
        "std_dev": 0.12,
    }
    pochmm = PocHMM()
    naive = pochmm.posterior_karyotypes(*pochmm.forward_backward(**kwargs)[::2])
    aware = pochmm.posterior_karyotypes(
        *pochmm.forward_backward(mcc=data["mcc"], **kwargs)[::2]
    )
    assert max(naive, key=naive.get) == "3p"
    assert max(aware, key=aware.get) == "1p"
    assert aware["1p"] > 0.95


# --- Genome-wide contamination estimation (est_mcc_genome / call_genome) --- #
def sim_genome(fraction, plan, m=500, use_lrr=True, delta=0.0, seed=20):
    """Simulate chromosomes sharing one contamination fraction, as per-chromosome lists."""
    chroms = [
        pgt_sim.full_ploidy_sim(
            m=m,
            ploidy=ploidy,
            mat_skew=skew,
            length=2e7,
            rec_rate=1e-8,
            std_dev=0.12,
            alpha=0.1,
            mcc=fraction,
            delta=delta,
            seed=seed + i,
        )
        for i, (ploidy, skew) in enumerate(plan)
    ]
    return {
        "bafs": [c["baf"] for c in chroms],
        "lrrs": [c["lrr"] if use_lrr else np.full(m, -9.0) for c in chroms],
        "sigmas": [c["sigmas"] if use_lrr else np.ones(m) for c in chroms],
        "pos": [c["pos"] for c in chroms],
        "haps": [c["mat_haps"] for c in chroms],
        "freqs": [c["af"] for c in chroms],
    }


# Three disomic chromosomes and one maternal trisomy.
genome_plan = [(2, 0.5), (2, 0.5), (3, 1.0), (2, 0.5)]


def test_genome_loglik_sums_chromosomes():
    """The genome log-likelihood is the sum of per-chromosome forward log-likelihoods."""
    data = sim_genome(0.1, genome_plan[:2])
    pochmm = PocHMM()
    total = pochmm.genome_loglik(mcc=0.1, delta=0.05, **data)
    parts = [
        pochmm.forward_algorithm(
            bafs=data["bafs"][i],
            lrrs=data["lrrs"][i],
            sigmas=data["sigmas"][i],
            pos=data["pos"][i],
            haps=data["haps"][i],
            freqs=data["freqs"][i],
            mcc=0.1,
            delta=0.05,
        )[4]
        for i in range(2)
    ]
    assert np.isclose(total, sum(parts))


@pytest.mark.parametrize("use_lrr", [True, False], ids=["with_lrr", "no_lrr"])
@pytest.mark.parametrize("fraction", [0.05, 0.15, 0.25])
def test_est_mcc_genome_recovers_truth(fraction, use_lrr):
    """The genome-wide estimate is close to the simulated fraction and lies inside its interval."""
    data = sim_genome(fraction, genome_plan, use_lrr=use_lrr)
    res = PocHMM().est_mcc_genome(**data)
    lo, hi = res["mcc_ci"]
    # BAF alone overestimates heavy contamination by ~0.01-0.02 at f = 0.25,
    # from inferring genotypes and phase without LRR, so it gets more slack.
    assert abs(res["mcc"] - fraction) < (0.03 if use_lrr else 0.05)
    assert lo <= res["mcc"] <= hi
    assert res["delta"] == 0.0


def test_est_mcc_genome_interval_coverage():
    """The 95% interval covers the simulated fraction in most replicates."""
    pochmm = PocHMM()
    covered = 0
    for seed in range(20, 60, 8):
        res = pochmm.est_mcc_genome(**sim_genome(0.15, genome_plan, seed=seed))
        covered += res["mcc_ci"][0] <= 0.15 <= res["mcc_ci"][1]
    # P(at least 4 of 5) is about 0.98 at the nominal 95% coverage.
    assert covered >= 4


def test_est_mcc_genome_uncontaminated():
    """An uncontaminated sample is estimated at zero contamination."""
    data = sim_genome(0.0, genome_plan)
    res = PocHMM().est_mcc_genome(**data)
    assert res["mcc"] == 0.0
    assert res["mcc_ci"][0] == 0.0


def test_est_mcc_genome_infers_offset():
    """With delta=None the LRR offset is estimated, and the contamination estimate is unaffected by it."""
    shifted = sim_genome(0.15, genome_plan, delta=0.2)
    pochmm = PocHMM()
    res = pochmm.est_mcc_genome(delta=None, **shifted)
    assert abs(res["delta"] - 0.2) < 0.03
    assert abs(res["mcc"] - 0.15) < 0.03
    assert res["mcc_ci"][0] <= 0.15 <= res["mcc_ci"][1]
    # Fixing the offset at zero when it is not biases the contamination estimate.
    fixed = pochmm.est_mcc_genome(delta=0.0, **shifted)
    assert fixed["mcc"] > res["mcc_ci"][1]


def test_est_delta_at_true_fraction():
    """The EM offset update recovers a simulated offset at the true contamination fraction."""
    data = sim_genome(0.1, genome_plan, delta=-0.15)
    delta, loglik = PocHMM().est_delta(mcc=0.1, delta=0.0, **data)
    assert abs(delta + 0.15) < 0.03
    assert np.isclose(loglik, PocHMM().genome_loglik(mcc=0.1, delta=delta, **data))


def test_call_genome_without_known_contamination():
    """call_genome rescues BAF-only calls at 30% MCC without being told the fraction."""
    plan = [(2, 0.5), (1, 1.0), (2, 0.5), (3, 0.0)]
    data = sim_genome(0.3, plan, m=1000, use_lrr=False)
    res = PocHMM().call_genome(**data)
    assert res["calls"] == ["2", "1p", "2", "3p"]
    assert abs(res["mcc"] - 0.3) < 0.03
    assert len(res["posteriors"]) == len(plan)

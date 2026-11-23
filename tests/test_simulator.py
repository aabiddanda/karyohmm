"""Test suite for simulation of PGT-A data."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from karyohmm import PGTSim, PGTSimMosaic, PGTSimSegmental, PGTSimVCF

pgt_sim = PGTSim()
pgt_sim_mosaic = PGTSimMosaic()
pgt_sim_segmental = PGTSimSegmental()


@given(
    length=st.floats(min_value=1e2, max_value=1e8),
    m=st.integers(min_value=1000, max_value=5000),
    ploidy=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=100, deadline=5000)
def test_pgt_sim_baf(length, m, ploidy):
    """Test for PGT simulations."""
    data = pgt_sim.full_ploidy_sim(m=m, ploidy=ploidy, length=length)
    assert data["m"] == m
    assert data["length"] == length
    assert np.max(data["pos"]) <= length
    assert "baf" in data.keys()
    if ploidy > 0:
        assert np.any(data["baf"] == 0.0) | np.any(data["baf"] == 1.0)


@given(
    length=st.floats(min_value=1e2, max_value=1e8),
    m=st.integers(min_value=1000, max_value=5000),
    ploidy=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=100, deadline=5000)
def test_pgt_sim_reads(length, m, ploidy):
    """Test for PGT simulations."""
    data = pgt_sim.full_ploidy_sim(m=m, reads=True, ploidy=ploidy, length=length)
    assert data["m"] == m
    assert data["length"] == length
    assert np.max(data["pos"]) <= length
    assert "alt_reads" in data.keys()
    assert "ref_reads" in data.keys()
    assert np.all(data["alt_reads"] >= 0)
    assert np.all(data["ref_reads"] >= 0)


@pytest.mark.parametrize(
    "mat_skew,ploidy,expected",
    [
        (0.9999, 1, "1p"),
        (1e-4, 1, "1m"),
        (0.9999, 3, "3m"),
        (1e-4, 3, "3p"),
        (0.9999, 2, "2"),
        (1e-4, 2, "2"),
    ],
)
def test_parental_origin_sim(mat_skew, ploidy, expected):
    """Test the appropriate parental-origin for whole-chromosome simulations."""
    data = pgt_sim.full_ploidy_sim(m=100, ploidy=ploidy, length=1e6, mat_skew=mat_skew)
    assert data["aploid"] == expected


@given(
    length=st.floats(min_value=1e2, max_value=1e8),
    m=st.integers(min_value=2, max_value=1000),
    nsib=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=20, deadline=5000)
def test_pgt_siblings(length, m, nsib):
    """Test for PGT simulations."""
    data = pgt_sim.sibling_euploid_sim(nsibs=nsib, m=m, length=length)
    assert data["m"] == m
    assert data["length"] == length
    assert data["nsibs"] == nsib
    assert np.max(data["pos"]) <= length
    for i in range(nsib):
        assert f"baf_embryo{i}" in data.keys()
        assert f"geno_embryo{i}" in data.keys()


@given(
    length=st.floats(min_value=1e2, max_value=1e8),
    m=st.integers(min_value=2, max_value=1000),
    nsib=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=20, deadline=5000)
def test_pgt_siblings_reads(length, m, nsib):
    """Test for PGT simulations."""
    data = pgt_sim.sibling_euploid_sim(nsibs=nsib, reads=True, m=m, length=length)
    assert data["m"] == m
    assert data["length"] == length
    assert data["nsibs"] == nsib
    assert np.max(data["pos"]) <= length
    for i in range(nsib):
        assert f"alt_reads{i}" in data.keys()
        assert f"ref_reads{i}" in data.keys()


@given(
    m=st.integers(min_value=1000, max_value=5000),
    ploidy=st.integers(min_value=0, max_value=3),
    frac_chrom=st.floats(min_value=0.1, max_value=0.5),
)
@settings(max_examples=10, deadline=5000)
def test_pgt_segmental(m, ploidy, frac_chrom):
    """Test for PGT segmental aneuploidy simulation."""
    mean_size = np.round(m * frac_chrom)
    data = pgt_sim_segmental.full_segmental_sim(m=m, ploidy=ploidy, mean_size=mean_size)
    assert data["m"] == m
    assert "baf" in data.keys()
    assert np.any(data["baf"] == 0.0) | np.any(data["baf"] == 1.0)
    if ploidy != 2:
        assert not np.all(data["ploidies"] == 2)
    else:
        assert np.all(data["ploidies"] == 2)


@pytest.mark.parametrize("mismap_rate", [0.0, 1e-2, 1e-1])
def test_mismap_rate_sets_background_depth(mismap_rate):
    """`mismap_rate` controls depth at nullisomic sites, independently of `eps`."""
    data = pgt_sim.full_ploidy_sim(
        m=4000,
        length=20e6,
        ploidy=0,
        reads=True,
        coverage=50.0,
        mismap_rate=mismap_rate,
    )
    depth = data["alt_reads"].astype(float) + data["ref_reads"]
    assert depth.mean() == pytest.approx(50.0 * mismap_rate / 2.0, abs=0.1)
    if mismap_rate == 0.0:
        assert np.all(depth == 0)


@pytest.mark.parametrize("eps", [1e-3, 5e-2])
def test_eps_leaves_depth_untouched(eps):
    """`eps` shifts allele fractions but must not change total depth."""
    data = pgt_sim.full_ploidy_sim(
        m=4000, length=20e6, ploidy=2, reads=True, coverage=50.0, eps=eps
    )
    alt = data["alt_reads"].astype(float)
    depth = alt + data["ref_reads"]
    geno = data["geno_embryo"]
    assert depth.mean() == pytest.approx(50.0 * (2 + 1e-2) / 2.0, abs=0.5)
    hom_ref = geno == 0
    assert alt[hom_ref].sum() / depth[hom_ref].sum() == pytest.approx(eps, abs=5e-3)


@given(
    m=st.integers(min_value=1000, max_value=5000),
    ploidy=st.integers(min_value=0, max_value=3),
    frac_chrom=st.floats(min_value=0.1, max_value=0.5),
)
@settings(max_examples=10, deadline=5000)
def test_pgt_segmental_reads(m, ploidy, frac_chrom):
    """Test for PGT segmental aneuploidy simulation with read counts."""
    mean_size = np.round(m * frac_chrom)
    data = pgt_sim_segmental.full_segmental_sim(
        m=m, reads=True, ploidy=ploidy, mean_size=mean_size, coverage=20.0
    )
    assert data["m"] == m
    assert data["baf"] is None
    assert np.all(data["alt_reads"] >= 0)
    assert np.all(data["ref_reads"] >= 0)
    # Depth must track the per-site copy number of the segment.
    depth = data["alt_reads"].astype(float) + data["ref_reads"]
    ploidies = data["ploidies"]
    means = {k: depth[ploidies == k].mean() for k in np.unique(ploidies)}
    assert sorted(means) == sorted(means, key=lambda k: means[k])


@pytest.mark.parametrize("aneu_type", ["0", "1m", "1p", "2", "3m", "3p"])
def test_segmental_paths_dosage_le_ploidy(aneu_type):
    """Embryo dosage can never exceed the ploidy, including at the first site."""
    rng = np.random.default_rng(0)
    m = 200
    mat_haps = rng.integers(0, 2, (2, m))
    pat_haps = rng.integers(0, 2, (2, m))
    pos = np.sort(rng.uniform(high=1e6, size=m))
    for start in [0, 1, 5]:
        out = pgt_sim_segmental.sim_haplotype_paths_segmental(
            mat_haps, pat_haps, pos, start=start, end=start + 20, aneu_type=aneu_type
        )
        mat_hap1, pat_hap1, ploidies = out[0], out[1], out[9]
        geno = mat_hap1.astype(int) + pat_hap1.astype(int)
        assert np.all(geno <= ploidies)


@pytest.mark.parametrize("reads", [True, False])
@pytest.mark.parametrize("ploidy", [0, 1, 2, 3])
def test_segmental_sim_from_haps(reads, ploidy):
    """Segmental simulation from known parental haplotypes runs in both modes."""
    rng = np.random.default_rng(0)
    m = 1000
    mat_haps = rng.integers(0, 2, (2, m))
    pat_haps = rng.integers(0, 2, (2, m))
    pos = np.sort(rng.uniform(high=10e6, size=m))
    data = pgt_sim_segmental.sim_from_haps(
        mat_haps, pat_haps, pos, reads=reads, ploidy=ploidy, mean_size=200
    )
    assert np.asarray(data["af"]).size == m
    if reads:
        assert data["baf"] is None
        assert np.all(data["alt_reads"] >= 0)
        assert np.all(data["ref_reads"] >= 0)
    else:
        assert data["baf"].size == m
        assert data["alt_reads"] is None
        assert data["ref_reads"] is None


@pytest.mark.parametrize(
    "mat_skew,ploidy,expected",
    [
        (0.9999, 1, "1p"),
        (1e-4, 1, "1m"),
        (0.9999, 3, "3m"),
        (1e-4, 3, "3p"),
        (0.9999, 2, "2"),
        (1e-4, 2, "2"),
    ],
)
def test_parental_origin_sim_segmental(mat_skew, ploidy, expected):
    """Test the appropriate parental-origin for segmental simulation."""
    data = pgt_sim_segmental.full_segmental_sim(
        m=2000, ploidy=ploidy, mean_size=400, mat_skew=mat_skew
    )
    assert data["seg_aneuploidy_type"] == expected


@pytest.fixture
def valid_vcf_file():
    """Small VCF file shipped with the project."""
    return "data/chr1.subset.1kg_phase3.phased.n50.vcf.gz"


def test_pgt_vcf(valid_vcf_file):
    """Test reading in a VCF and outputting key parameters."""
    pgt_vcf = PGTSimVCF()
    mat_haps, pat_haps, pos, afs = pgt_vcf.gen_parental_haplotypes(
        vcf_fp=valid_vcf_file,
        maternal_id="HG00096",
        paternal_id="HG00107",
        gts012=True,
        threads=4,
    )
    assert mat_haps.ndim == 2
    assert pat_haps.ndim == 2
    assert mat_haps.size == pat_haps.size
    assert mat_haps.shape[0] == 2
    assert pat_haps.shape[0] == 2
    assert mat_haps.shape[1] == pos.size
    assert pat_haps.shape[1] == pos.size
    assert afs.size == pos.size
    assert np.all(np.isin(mat_haps, [0, 1]))
    assert np.all(np.isin(pat_haps, [0, 1]))


def test_pgt_sim_from_vcf(valid_vcf_file):
    """Test actually simulating from pre-existing haplotypes."""
    pgt_vcf = PGTSimVCF()
    pgt_sim = PGTSim()
    mat_haps, pat_haps, pos, afs = pgt_vcf.gen_parental_haplotypes(
        vcf_fp=valid_vcf_file,
        maternal_id="HG00096",
        paternal_id="HG00107",
        gts012=True,
        threads=4,
    )
    results = pgt_sim.sim_from_haps(mat_haps=mat_haps, pat_haps=pat_haps, pos=pos)
    assert "baf" in results
    assert "alt_reads" in results
    assert "ref_reads" in results
    assert results["alt_reads"] is None
    assert results["ref_reads"] is None


@given(
    length=st.floats(min_value=1e2, max_value=1e8),
    m=st.integers(min_value=100, max_value=3000),
    k=st.integers(min_value=2, max_value=20),
)
@settings(max_examples=10, deadline=5000)
def test_pgt_sim_ref_panel(length, m, k):
    """Test for PGT simulations."""
    data = pgt_sim.full_ploidy_sim(m=m, ploidy=2, length=length)
    assert data["m"] == m
    assert data["length"] == length
    assert np.max(data["pos"]) <= length
    assert "baf" in data.keys()
    ref_panel = pgt_sim.sim_haplotype_ref_panel(
        haps=data["mat_haps"], pos=data["pos"], panel_size=k
    )
    assert ref_panel.ndim == 2
    assert ref_panel.shape[0] == k
    assert ref_panel.shape[1] == m


@given(
    length=st.floats(min_value=1e2, max_value=1e8),
    m=st.integers(min_value=100, max_value=1000),
    cc=st.floats(min_value=0.0, max_value=0.5),
)
@settings(max_examples=10, deadline=5000)
def test_sim_mcc(length, m, cc):
    """Test for PGT simulations."""
    data = pgt_sim.full_ploidy_sim(m=m, ploidy=2, length=length)
    assert data["m"] == m
    assert data["length"] == length
    assert np.max(data["pos"]) <= length
    assert "baf" in data.keys()
    cc_baf = pgt_sim.sim_cell_contamination(
        baf=data["baf"], haps=data["mat_haps"], fraction=cc, seed=42
    )
    baf = data["baf"]
    idx = (baf != 0) & (baf != 1)
    if ~np.isclose(cc, 0):
        assert np.all(cc_baf[idx] != baf[idx])


@given(
    length=st.floats(min_value=1e2, max_value=1e8),
    m=st.integers(min_value=100, max_value=1000),
    c=st.sampled_from(["2p0", "2p1", "2m0", "2m1"]),
)
@settings(deadline=None)
def test_pgt_sim_upd(length, m, c):
    data = pgt_sim.full_ploidy_sim(m=m, length=length, ploidy=2, upd=c)
    assert data["m"] == m
    assert data["length"] == length
    assert np.max(data["pos"]) <= length
    assert "baf" in data.keys()
    assert "lrr" in data.keys()
    if c in ["2p0", "2m0"]:
        # isodisomy so no true hets
        assert np.all((data["geno_embryo"] == 0) | (data["geno_embryo"] == 2))


# ---------------------------------------------------------------------------
# PGTSimMosaic tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ploidies,props,ncells",
    [
        (np.array([2]), np.array([1.0]), 5),
        (np.array([1, 2]), np.array([0.3, 0.7]), 10),
        (np.array([2, 3]), np.array([0.6, 0.4]), 10),
        (np.array([1, 2, 3]), np.array([0.2, 0.6, 0.2]), 15),
    ],
)
def test_mosaic_sim_mixed_ploidy(ploidies, props, ncells):
    """PGTSimMosaic.mixed_ploidy_sim returns expected keys and shapes."""
    data = pgt_sim_mosaic.mixed_ploidy_sim(
        ploidies=ploidies, props=props, ncells=ncells, m=500, length=1e6, seed=42
    )
    assert data["m"] == 500
    assert "baf" in data
    assert "lrr" in data
    assert "sigmas" in data
    assert "pos" in data
    assert "mat_haps" in data
    assert "pat_haps" in data
    assert data["baf"].shape == (500,)
    assert data["lrr"].shape == (500,)
    assert np.all(data["baf"] >= 0) and np.all(data["baf"] <= 1)
    assert np.max(data["pos"]) <= 1e6
    # aploid list has one entry per unique ploidy actually sampled
    assert len(data["aploid"]) == np.unique(data["ploidies"]).size


@pytest.mark.parametrize(
    "props",
    [
        np.array([0.0, 1.0, 0.0, 0.0]),
        np.array([0.0, 0.3, 0.7, 0.0]),
        np.array([0.0, 0.0, 0.6, 0.4]),
    ],
)
def test_mosaic_sim_disomy_dominant(props):
    """mixed_ploidy_sim with high disomy proportion produces near-disomy BAF."""
    ploidies = np.array([0, 1, 2, 3])
    data = pgt_sim_mosaic.mixed_ploidy_sim(
        ploidies=ploidies,
        props=props,
        ncells=20,
        m=1000,
        length=5e6,
        std_dev=0.05,
        seed=1,
    )
    assert data["m"] == 1000
    assert "baf" in data
    assert data["baf"].shape == (1000,)


def test_mosaic_sim_reproducible():
    """Two runs with the same seed produce identical output."""
    kwargs = dict(
        ploidies=np.array([2, 3]),
        props=np.array([0.6, 0.4]),
        ncells=10,
        m=500,
        length=1e6,
        seed=99,
    )
    d1 = pgt_sim_mosaic.mixed_ploidy_sim(**kwargs)
    d2 = pgt_sim_mosaic.mixed_ploidy_sim(**kwargs)
    np.testing.assert_array_equal(d1["baf"], d2["baf"])
    np.testing.assert_array_equal(d1["pos"], d2["pos"])


def test_mosaic_sim_from_haps():
    """PGTSimMosaic.sim_from_haps works with pre-computed parental haplotypes."""
    base = pgt_sim.full_ploidy_sim(m=500, ploidy=2, length=1e6, seed=7)
    data = pgt_sim_mosaic.sim_from_haps(
        mat_haps=base["mat_haps"],
        pat_haps=base["pat_haps"],
        pos=base["pos"],
        ploidies=np.array([2, 3]),
        props=np.array([0.7, 0.3]),
        ncells=8,
        seed=7,
    )
    assert "baf" in data
    assert "lrr" in data
    assert data["baf"].shape == (500,)
    assert np.all(data["baf"] >= 0) and np.all(data["baf"] <= 1)


def test_mosaic_sim_props_normalization():
    """mixed_ploidy_sim normalizes props that do not sum to 1."""
    data = pgt_sim_mosaic.mixed_ploidy_sim(
        ploidies=np.array([2, 3]),
        props=np.array([3.0, 1.0]),
        ncells=10,
        m=300,
        length=5e5,
        seed=5,
    )
    assert data["m"] == 300
    assert "baf" in data


@pytest.mark.parametrize("seed", [1, 2, 3, 42])
def test_mosaic_sim_bulk_shapes(seed):
    """Bulk BAF/LRR/sigmas arrays have shape (ncells, m)."""
    ncells = 6
    m = 400
    data = pgt_sim_mosaic.mixed_ploidy_sim(
        ploidies=np.array([2, 3]),
        props=np.array([0.5, 0.5]),
        ncells=ncells,
        m=m,
        length=2e6,
        seed=seed,
    )
    assert data["baf_embryo_bulk"].shape == (ncells, m)
    assert data["lrr_embryo_bulk"].shape == (ncells, m)
    assert data["sigmas_embryo_bulk"].shape == (ncells, m)


@pytest.mark.parametrize("mcc", [0.05, 0.2, 0.4])
def test_sim_mcc_baf_disomy(mcc):
    """Contaminated disomic BAF sits at 0.5 +/- mcc / 2 at mother-hom, embryo-het sites."""
    data = pgt_sim.full_ploidy_sim(m=5000, ploidy=2, std_dev=0.03, mcc=mcc, seed=7)
    mg = data["mat_haps"].sum(axis=0)
    het = data["geno_embryo"] == 1
    assert data["mcc"] == mcc
    assert np.isclose(np.mean(data["baf"][het & (mg == 2)]), 0.5 + mcc / 2, atol=5e-3)
    assert np.isclose(np.mean(data["baf"][het & (mg == 0)]), 0.5 - mcc / 2, atol=5e-3)
    # Point masses survive only where embryo and mother share a homozygous genotype.
    assert np.all(data["baf"][(data["geno_embryo"] == 0) & (mg > 0)] > 0)
    assert np.any(data["baf"][(data["geno_embryo"] == 0) & (mg == 0)] == 0)


@pytest.mark.parametrize("mcc", [0.1, 0.3])
@pytest.mark.parametrize("ploidy", [1, 3])
def test_sim_mcc_copy_weighted(mcc, ploidy):
    """Contamination mixes allele copies, weighting the embryo by its copy number."""
    data = pgt_sim.full_ploidy_sim(
        m=5000, ploidy=ploidy, std_dev=0.03, mcc=mcc, alpha=0.1, seed=11
    )
    mg = data["mat_haps"].sum(axis=0)
    g = data["geno_embryo"]
    # Which embryo dosages occur at mother-hom sites depends on the parent of origin.
    genos = [x for x in range(ploidy + 1) if np.sum((g == x) & (mg == 2)) > 100]
    assert len(genos) > 0
    for geno in genos:
        idx = (g == geno) & (mg == 2)
        mu = ((1 - mcc) * geno + mcc * 2) / ((1 - mcc) * ploidy + 2 * mcc)
        if mu < 1:
            assert np.isclose(np.mean(data["baf"][idx]), mu, atol=5e-3)
    lrr_mu = np.log2(((1 - mcc) * ploidy + 2 * mcc) / 2)
    assert np.isclose(np.mean(data["lrr"]), lrr_mu, atol=0.02)


@pytest.mark.parametrize("ploidy", [1, 3])
def test_sim_mcc_segmental(ploidy):
    """Segmental simulations carry LRR, contaminated inside and outside the segment."""
    mcc = 0.2
    data = pgt_sim_segmental.full_segmental_sim(
        m=4000, ploidy=ploidy, mean_size=1000, alpha=0.1, mcc=mcc, seed=3
    )
    ploidies = data["ploidies"]
    assert data["lrr"].size == data["sigmas"].size == 4000
    for k in np.unique(ploidies):
        mu = np.log2(((1 - mcc) * k + 2 * mcc) / 2)
        assert np.isclose(np.mean(data["lrr"][ploidies == k]), mu, atol=0.03)
    clean = pgt_sim_segmental.full_segmental_sim(
        m=4000, ploidy=ploidy, mean_size=1000, alpha=0.1, seed=3
    )
    assert np.array_equal(clean["ploidies"], ploidies)
    assert np.isclose(np.mean(clean["lrr"][ploidies == 2]), 0.0, atol=0.02)


def test_sim_mcc_reads_not_implemented():
    """Contamination is only simulated for array data."""
    with pytest.raises(NotImplementedError):
        pgt_sim.full_ploidy_sim(m=100, reads=True, mcc=0.1)
    with pytest.raises(NotImplementedError):
        pgt_sim_segmental.full_segmental_sim(m=100, reads=True, mcc=0.1)


@pytest.mark.parametrize("fraction", [0.0, 0.1, 0.3])
def test_sim_cell_contamination_copy_weighted(fraction):
    """Post-hoc contamination reduces to the BAF mixture at disomy and copy-weights otherwise."""
    data = pgt_sim.full_ploidy_sim(m=500, ploidy=2, seed=9)
    baf, haps = data["baf"], data["mat_haps"]
    mg = haps.sum(axis=0)
    disomy = pgt_sim.sim_cell_contamination(baf, haps, fraction=fraction)
    assert np.allclose(disomy, (1 - fraction) * baf + fraction * mg / 2)
    trisomy = pgt_sim.sim_cell_contamination(baf, haps, fraction=fraction, ploidy=3)
    expected = ((1 - fraction) * 3 * baf + fraction * mg) / (
        (1 - fraction) * 3 + 2 * fraction
    )
    assert np.allclose(trisomy, expected)
    lrr = pgt_sim.sim_cell_contamination_lrr(np.full(3, np.log2(1.5)), fraction=fraction)
    assert np.allclose(lrr, np.log2(((1 - fraction) * 3 + 2 * fraction) / 2))


@pytest.mark.parametrize("mcc", [0.0, 0.2])
def test_sim_lrr_offset_triploidy(mcc):
    """An LRR offset of the triploid copy ratio re-centres a trisomy on zero, as arrays do."""
    delta = np.log2(((1 - mcc) * 3 + 2 * mcc) / 2)
    raw = pgt_sim.full_ploidy_sim(m=4000, ploidy=3, alpha=0.1, mcc=mcc, seed=5)
    normed = pgt_sim.full_ploidy_sim(
        m=4000, ploidy=3, alpha=0.1, mcc=mcc, delta=delta, seed=5
    )
    assert normed["delta"] == delta
    assert np.allclose(normed["lrr"], raw["lrr"] - delta)
    assert np.array_equal(normed["baf"], raw["baf"])
    assert np.isclose(np.mean(normed["lrr"]), 0.0, atol=0.02)
    seg = pgt_sim_segmental.full_segmental_sim(
        m=2000, ploidy=3, mean_size=500, alpha=0.1, mcc=mcc, seed=5
    )
    seg_normed = pgt_sim_segmental.full_segmental_sim(
        m=2000, ploidy=3, mean_size=500, alpha=0.1, mcc=mcc, delta=delta, seed=5
    )
    assert np.allclose(seg_normed["lrr"], seg["lrr"] - delta)


def test_sim_starting_haplotypes_random():
    """Duplicated copies start on a random haplotype, and heterodisomy on both homologues."""
    starts = []
    for seed in range(1, 21):
        data = pgt_sim.full_ploidy_sim(m=200, ploidy=3, mat_skew=1.0, seed=seed)
        starts.append(data["zs_maternal"][:, 0])
        upd = pgt_sim.full_ploidy_sim(m=200, ploidy=2, upd="2m1", seed=seed)
        assert upd["zs_maternal"][0, 0] != upd["zs_maternal"][1, 0]
    starts = np.array(starts)
    assert 0 < starts.mean() < 1
    assert np.any(starts[:, 0] != starts[:, 1])

"""
Karyohmm: HMM-based inference of chromosomal copy-number state from array intensity data.

Karyohmm implements haplotype-aware hidden Markov models for detecting aneuploidies,
mosaic copy-number changes, and maternal-cell contamination from B-allele frequency
(BAF) and log-R ratio (LRR) data when parental genotypes are available.  All models
condition on phased parental haplotypes and exploit Mendelian transmission to
distinguish true copy-number signal from genotyping noise.

Available classes
-----------------
- :class:`MetaHMM`: Whole-chromosome aneuploidy detection across a full ploidy state space
  (nullisomy, monosomy, disomy, trisomy, and optional UPD states).
- :class:`QuadHMM`: Joint HMM for two sibling embryos; used to call inter-sibling haplotype
  sharing and to detect crossover recombination events (Roach et al. 2010).
- :class:`PocHMM`: Aneuploidy inference in products-of-conception with only one observed parent
  (mother–child duo or father–child duo); marginalises over the unobserved parent using
  population allele frequencies.
- :class:`MccEst`: Maximum-likelihood estimator of maternal-cell contamination (MCC) fraction
  from BAF data, with both unphased and phase-aware HMM variants and genome-wide
  multi-chromosome support.
- :class:`MosaicEst`: Estimator of mosaic cell fraction from BAF imbalance at heterozygous
  sites using a 3-state (gain / neutral / loss) HMM (Loh et al. model).
- :class:`PhaseCorrect`: Mendelian phase correction for parental haplotypes using BAF data
  from multiple sibling embryos.
- :class:`RecombEst`: Simplified crossover detection following Coop et al. 2007, extended
  with QuadHMM-based interval refinement.

"""

import warnings

import numpy as np
from karyohmm_utils import (
    backward_algo,
    backward_algo_duo,
    backward_algo_reads,
    backward_algo_sibs,
    emission_baf,
    emission_lrr,
    forward_algo,
    forward_algo_duo,
    forward_algo_reads,
    forward_algo_sibs,
    forward_mcc_phased_poc,
    forward_mcc_phased_trio,
    logaddexp,
    logsumexp,
    loglik_mcc,
    loglik_mcc_phased,
    mat_dosage,
    norm_logl,
    pat_dosage,
    viterbi_algo,
    viterbi_algo_reads,
    viterbi_algo_sibs,
)
from scipy.optimize import brentq, minimize, minimize_scalar
from scipy.special import logsumexp as logsumexp_sp
from scipy.stats import chi2, truncnorm


class AneuploidyHMM:
    """Abstract base class shared by all karyohmm HMM models.

    Defines the common interface (``forward_algorithm``, ``backward_algorithm``,
    ``forward_backward``, ``viterbi_algorithm``) and shared parameter-estimation
    helpers.  Concrete subclasses (:class:`MetaHMM`, :class:`QuadHMM`,
    :class:`PocHMM`) implement the state space and emission model appropriate for
    their data type.

    Attributes
    ----------
    ploidy : int
        Expected ploidy of the embryo (0 = meta/unknown, 2 = disomy).
    aploid : str or None
        String tag identifying the active model variant
        (e.g. ``"meta"``, ``"disomy"``, ``"meta+upd"``).
    """

    def __init__(self):
        """Initialize the base aneuploidy HMM class."""
        self.ploidy = 2
        self.aploid = None

    def get_state_str(self, state):
        """Obtain a string representation of the HMM state.

        Parameters
        ----------
        state : tuple
            A tuple of integers (typically length 4) encoding haplotype indices.

        Returns
        -------
        state_str : str
            String representation of the state.

        """
        t = []
        for i, s in enumerate(state):
            if s != -1:
                if i < 2:
                    t.append(f"m{s}")
                else:
                    t.append(f"p{s}")
        if not t:
            t.append("0")
        return "".join(t)

    # Per-algorithm tolerance keys recognized by scipy.optimize.minimize.
    # Nelder-Mead uses xatol/fatol; L-BFGS-B uses ftol/gtol; Powell uses xtol/ftol.
    _ALGO_DEFAULT_OPTIONS = {
        "Nelder-Mead": {"xatol": 1e-4, "fatol": 1e-4, "disp": False},
        "L-BFGS-B": {"ftol": 1e-4, "gtol": 1e-5, "disp": False},
        "Powell": {"xtol": 1e-4, "ftol": 1e-4, "disp": False},
    }

    def est_sigma_pi0(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        algo="L-BFGS-B",
        pi0_bounds=(0.01, 0.99),
        sigma_bounds=(1e-2, 1.0),
        opt_tol=1e-4,
        opt_options=None,
        **kwargs,
    ):
        """Estimate sigma and pi0 under the B-Allele Frequency model using optimization of forward algorithm likelihood.

        Parameters
        ----------
        bafs : np.ndarray
            B-allele frequencies across all m sites.
        pos : np.ndarray
            Basepair positions of the SNPs.
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        algo : str
            One of Nelder-Mead, L-BFGS-B, or Powell algorithms for optimization.
        opt_tol : float
            Master tolerance passed to scipy minimize (default 1e-4).
        opt_options : dict or None
            Algorithm-specific options passed to scipy minimize; merged over
            per-algorithm defaults, so only keys you want to override are needed.

        Returns
        -------
        pi0_est : float
            Estimate of sparsity parameter (pi0) for B-allele emission model.
        sigma_est : float
            Estimate of noise parameter (sigma) for B-allele emission model.

        """
        assert algo in ["Nelder-Mead", "L-BFGS-B", "Powell"]
        assert (len(pi0_bounds) == 2) and (len(sigma_bounds) == 2)
        assert (pi0_bounds[0] > 0) and (pi0_bounds[1] > 0)
        assert (pi0_bounds[0] < 1) and (pi0_bounds[1] < 1)
        assert pi0_bounds[0] < pi0_bounds[1]
        assert (sigma_bounds[0] > 0) and (sigma_bounds[1] > 0)
        assert sigma_bounds[0] < sigma_bounds[1]
        mid_pi0 = np.mean(pi0_bounds)
        mid_sigma = np.mean(sigma_bounds)
        options = {**self._ALGO_DEFAULT_OPTIONS[algo], **(opt_options or {})}
        opt_res = minimize(
            lambda x: (
                -self.forward_algorithm(
                    bafs=bafs,
                    pos=pos,
                    mat_haps=mat_haps,
                    pat_haps=pat_haps,
                    pi0=x[0],
                    std_dev=x[1],
                    **kwargs,
                )[4]
            ),
            x0=[mid_pi0, mid_sigma],
            method=algo,
            bounds=[pi0_bounds, sigma_bounds],
            tol=opt_tol,
            options=options,
        )
        pi0_est = opt_res.x[0]
        sigma_est = opt_res.x[1]
        return pi0_est, sigma_est

    def est_sigma_pi0_baf(
        self,
        bafs,
        mus=(0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0),
        atom_tol=0.0,
        max_iter=100,
        tol=1e-6,
    ):
        """Estimate sigma and pi0 from B-allele frequencies alone, without the HMM.

        Requires no haplotypes, reference panel, or allele frequencies, and runs
        roughly twenty times faster than :meth:`est_sigma_pi0`.

        A continuous density places zero probability on an exact point, so B-allele
        frequencies equal to 0.0 or 1.0 can only be point-mass draws and their
        frequency is available in closed form.  The remaining sites form a
        shared-scale truncated-normal mixture over the dosage grid, which spans
        monosomy, disomy and trisomy so no karyotype is presupposed.  Estimating the
        point masses separately from the scale is what avoids the sigma/pi0
        trade-off that biases the likelihood-based estimator when it is handed
        imperfect allele frequencies.

        Fit by expectation-maximization.  A single-shot optimizer over the same
        likelihood is not reliable here: it settles on degenerate optima, and
        without a bound on sigma it diverges on monosomy, which has no
        heterozygous band to pin the scale.

        Parameters
        ----------
        bafs : np.ndarray
            B-allele frequencies across all m sites.
        mus : tuple
            Dosage-grid means for the continuous mixture components.
        atom_tol : float
            Tolerance for treating a B-allele frequency as an exact 0/1 point
            mass.  The default of 0.0 tests exact equality, which suits simulated
            data; measured arrays need a small window such as 1e-4.
        max_iter : int
            Maximum number of EM iterations.
        tol : float
            Relative log-likelihood convergence threshold.

        Returns
        -------
        pi0_est : float
            Estimate of sparsity parameter (pi0) for B-allele emission model.
        sigma_est : float
            Estimate of noise parameter (sigma) for B-allele emission model.

        Notes
        -----
        Sigma is only weakly identifiable above roughly 0.3: a normal truncated to
        [0, 1] tends toward uniform as the scale grows, so the likelihood flattens
        and estimates saturate near 0.28 regardless of the true value.  This is a
        property of the emission model rather than of this estimator.

        """
        b = np.asarray(bafs, dtype=float)
        is0 = (b <= atom_tol) if atom_tol > 0 else (b == 0.0)
        is1 = (b >= 1.0 - atom_tol) if atom_tol > 0 else (b == 1.0)
        x = b[~(is0 | is1)]
        if x.size < 10:
            return np.nan, np.nan
        a0, a1 = is0.sum() / b.size, is1.sum() / b.size
        mu = np.asarray(mus, dtype=float)[:, None]

        def logpdf(sigma):
            return truncnorm.logpdf(
                x[None, :], (0.0 - mu) / sigma, (1.0 - mu) / sigma, loc=mu, scale=sigma
            )

        logw = np.full((mu.size, 1), -np.log(mu.size))
        sigma_est, prev = max(x.std(), 0.02), -np.inf
        for _ in range(max_iter):
            comp = logw + logpdf(sigma_est)
            denom = logsumexp_sp(comp, axis=0)
            resp = np.exp(comp - denom)
            w = np.maximum(resp.sum(axis=1), 1e-12)
            logw = np.log(w / w.sum())[:, None]
            sigma_est = float(
                minimize_scalar(
                    lambda s: -(resp * logpdf(s)).sum(),
                    bounds=(0.01, 0.6),
                    method="bounded",
                ).x
            )
            cur = float(denom.sum())
            if abs(cur - prev) < tol * max(1.0, abs(prev)):
                break
            prev = cur

        # pi0 is conditional on the site being homozygous in expectation, so weigh
        # the point masses only against the mu=0 and mu=1 continuous components.
        w = np.exp(logw.ravel())
        hom = a0 + a1 + (1.0 - a0 - a1) * (w[0] + w[-1])
        pi0_est = (a0 + a1) / hom if hom > 0 else np.nan
        return float(pi0_est), sigma_est

    def full_param_inference(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        algo="L-BFGS-B",
        pi0_bounds=(0.01, 0.99),
        sigma_bounds=(1e-2, 1.0),
    ):
        """Full parameter inference under the B-allele frequency model using naive optimization of the forward algorithm.

        Parameters
        ----------
        bafs : np.ndarray
            B-allele frequencies across all m sites.
        pos : np.ndarray
            Basepair positions of the SNPs.
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        algo : str
            One of Nelder-Mead, L-BFGS-B, or Powell algorithms for optimization.

        Notes
        -----
        Not implemented yet.

        """
        raise NotImplementedError(
            "We have not implemented full joint parameter inference yet!"
        )


class MetaHMM(AneuploidyHMM):
    """HMM for whole-chromosome aneuploidy detection in trio PGT data.

    Evaluates a joint state space covering nullisomy (0), maternal and paternal
    monosomy (1m / 1p), disomy (2), maternal and paternal trisomy (3m / 3p),
    and optionally uniparental disomy (UPD) states.  Each hidden state encodes
    which pair of parental haplotypes were transmitted to the embryo; transitions
    follow an exponential kernel parameterised by an intra-karyotype recombination
    rate ``r`` and an inter-karyotype switching rate ``a``.

    The BAF emission at each site is a Gaussian mixture centred on the expected
    allele dosage for the current copying state, controlled by ``pi0`` (fraction
    of sites with no signal) and ``std_dev`` (noise).  LRR data are incorporated
    through a Gaussian emission centred on the expected copy-number fold change.

    Parameters
    ----------
    disomy : bool, optional
        Restrict the state space to disomy-only (4 states).  Useful for genotyping
        embryos without calling aneuploidies.  Default ``False``.
    upd : bool, optional
        Add uniparental disomy states to the full state space.  Requires LRR data
        to distinguish UPD from normal disomy.  Default ``False``.

    Attributes
    ----------
    states : list of tuple
        Active HMM states; each tuple is ``(m0, m1, p0, p1)`` where values are
        haplotype indices (0 or 1) or -1 when that haplotype is absent.
    karyotypes : np.ndarray of str
        Karyotype label for each state (e.g. ``"2"``, ``"3m"``, ``"1p"``).
    aploid : str
        One of ``"meta"``, ``"disomy"``, or ``"meta+upd"``.
    """

    def __init__(self, disomy=False, upd=False):
        """Initialize the MetaHMM class for determining chromosomal aneuploidy.

        Parameters
        ----------
        disomy : bool
            Assume disomy to limit the model state space.
        upd : bool
            Add uniparental disomy states to the full state space.

        """
        super().__init__()
        self.ploidy = 0
        self.aploid = "meta"
        self.nullisomy_state = [(-1, -1, -1, -1)]
        self.p_monosomy_states = [(-1, -1, 1, -1), (-1, -1, 0, -1)]
        self.m_monosomy_states = [(0, -1, -1, -1), (1, -1, -1, -1)]
        self.disomy_states = [
            (0, -1, 0, -1),
            (0, -1, 1, -1),
            (1, -1, 0, -1),
            (1, -1, 1, -1),
        ]
        # First two are the female states ...
        self.m_trisomy_states = [
            (0, 0, 0, -1),
            (0, 1, 0, -1),
            (1, 1, 0, -1),
            (0, 0, 1, -1),
            (0, 1, 1, -1),
            (1, 1, 1, -1),
        ]
        self.p_trisomy_states = [
            (0, -1, 0, 0),
            (1, -1, 0, 0),
            (0, -1, 0, 1),
            (1, -1, 0, 1),
            (0, -1, 1, 1),
            (1, -1, 1, 1),
        ]
        self.m_upd_states = [
            (0, 0, -1, -1),
            (1, 1, -1, -1),
            (0, 1, -1, -1),
            (1, 0, -1, -1),
        ]
        self.p_upd_states = [
            (-1, -1, 0, 0),
            (-1, -1, 1, 1),
            (-1, -1, 0, 1),
            (-1, -1, 1, 0),
        ]
        if disomy:
            self.aploid = "disomy"
            self.states = self.disomy_states
            self.karyotypes = np.array(["2", "2", "2", "2"], dtype=str)
        else:
            if upd:
                self.aploid = "meta+upd"
                self.states = (
                    self.nullisomy_state
                    + self.m_monosomy_states
                    + self.p_monosomy_states
                    + self.disomy_states
                    + self.m_trisomy_states
                    + self.p_trisomy_states
                    + self.p_upd_states
                    + self.m_upd_states
                )
                self.karyotypes = np.array(
                    [
                        "0",
                        "1m",
                        "1m",
                        "1p",
                        "1p",
                        "2",
                        "2",
                        "2",
                        "2",
                        "3m",
                        "3m",
                        "3m",
                        "3m",
                        "3m",
                        "3m",
                        "3p",
                        "3p",
                        "3p",
                        "3p",
                        "3p",
                        "3p",
                        "2p0",
                        "2p0",
                        "2p1",
                        "2p1",
                        "2m0",
                        "2m0",
                        "2m1",
                        "2m1",
                    ],
                    dtype=str,
                )
            else:
                self.states = (
                    self.nullisomy_state
                    + self.m_monosomy_states
                    + self.p_monosomy_states
                    + self.disomy_states
                    + self.m_trisomy_states
                    + self.p_trisomy_states
                )
                self.karyotypes = np.array(
                    [
                        "0",
                        "1m",
                        "1m",
                        "1p",
                        "1p",
                        "2",
                        "2",
                        "2",
                        "2",
                        "3m",
                        "3m",
                        "3m",
                        "3m",
                        "3m",
                        "3m",
                        "3p",
                        "3p",
                        "3p",
                        "3p",
                        "3p",
                        "3p",
                    ],
                    dtype=str,
                )

    def warn_missing_lrr(self, lrrs):
        """Warn when UPD states are active but LRR data is absent."""
        if self.aploid == "meta+upd" and np.all(lrrs == -9.0):
            warnings.warn(
                "All LRR values are missing (sentinel -9.0) but UPD states are "
                "included in the model. UPD states share copy number 2 with disomy "
                "and rely primarily on BAF patterns when LRR is unavailable, which "
                "may reduce power to distinguish UPD from normal disomy.",
                UserWarning,
                stacklevel=3,
            )

    def check_read_inputs(
        self,
        alt_reads,
        ref_reads,
        pos,
        mat_haps,
        pat_haps,
        depths,
        eps,
        coverage,
        mismap_rate,
        r,
        a,
    ):
        """Validate the arguments shared by the three read-level engines.

        Kept in one place so the forward, backward and Viterbi read engines
        cannot come to disagree about what constitutes valid input.
        """
        assert alt_reads.ndim == 1
        assert ref_reads.ndim == 1
        assert alt_reads.size == ref_reads.size
        assert pos.ndim == 1
        assert (mat_haps.ndim == 2) & (pat_haps.ndim == 2)
        assert alt_reads.size == mat_haps.shape[1]
        assert alt_reads.size == pos.size
        assert mat_haps.shape == pat_haps.shape
        assert np.all(pos[1:] > pos[:-1])
        assert r < 0.5 and r > 0
        assert a < 0.5 and a > 0
        assert (eps > 0) & (eps <= 0.5)
        assert coverage > 0
        assert mismap_rate >= 0
        if depths is not None:
            assert depths.ndim == 1
            assert depths.size == alt_reads.size

    def forward_algorithm(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        mat_haps,
        pat_haps,
        pi0=0.5,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
        mat_err=0.0,
        pat_err=0.0,
    ):
        """Forward HMM algorithm under a multi-ploidy model.

        Parameters
        ----------
        bafs : np.ndarray
            B-allele frequencies across all m sites.
        lrrs : np.ndarray
            m-length array of log-R ratio values.
        sigmas : np.ndarray
            m-length array of per-site LRR noise standard deviations.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        pi0 : float
            Sparsity parameter for B-allele emission model.
        std_dev : float
            Standard deviation for B-allele emission model.
        r : float
            Intra-karyotype transition rate (recombination).
        a : float
            Inter-karyotype transition rate.
        unphased : bool
            Run the model in unphased mode.
        mat_err : float
            Per-site maternal genotyping error rate.
        pat_err : float
            Per-site paternal genotyping error rate.

        Returns
        -------
        alphas : np.ndarray
            Forward variable from HMM across k states.
        scaler : np.ndarray
            m-length array of scale parameters.
        states : list
            Tuple representation of states.
        karyotypes : np.ndarray
            Array of karyotypes in the MetaHMM model.
        loglik : float
            Total log-likelihood of B-allele frequency.

        """
        assert bafs.ndim == 1
        assert lrrs.ndim == 1
        assert sigmas.ndim == 1
        assert bafs.size == lrrs.size
        assert sigmas.size == lrrs.size
        assert pos.ndim == 1
        assert (mat_haps.ndim == 2) & (pat_haps.ndim == 2)
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert bafs.size == mat_haps.shape[1]
        assert bafs.size == pos.size
        assert mat_haps.shape == pat_haps.shape
        assert np.all(pos[1:] > pos[:-1])
        assert r < 0.5 and r > 0
        assert a < 0.5 and a > 0
        assert 0.0 <= mat_err < 0.5
        assert 0.0 <= pat_err < 0.5
        alphas, scaler, _, _, loglik = forward_algo(
            bafs,
            lrrs,
            sigmas,
            pos,
            mat_haps,
            pat_haps,
            self.states,
            self.karyotypes,
            r=r,
            a=a,
            pi0=pi0,
            std_dev=std_dev,
            unphased=int(unphased),
            mat_err=mat_err,
            pat_err=pat_err,
        )
        return alphas, scaler, self.states, self.karyotypes, loglik

    def forward_algorithm_reads(
        self,
        alt_reads,
        ref_reads,
        pos,
        mat_haps,
        pat_haps,
        depths=None,
        eps=1e-2,
        coverage=1.0,
        mismap_rate=1e-2,
        r=1e-8,
        a=1e-2,
        unphased=False,
    ):
        """Forward HMM algorithm for read-level (sequencing) data.

        Read-based counterpart of :meth:`forward_algorithm`: the state space and
        transition model are identical, and only the emission changes.  Allelic
        balance takes the role of the BAF emission and total depth the role of
        the LRR emission.  The two data types are never mixed -- BAF is usually
        derived from the same reads, so combining them double-counts the data.

        Parameters
        ----------
        alt_reads : np.ndarray
            m-length array of alternate-allele read counts.
        ref_reads : np.ndarray
            m-length array of reference-allele read counts.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        depths : np.ndarray, optional
            m-length array of total read depths, defaulting to
            ``alt_reads + ref_reads``.  Supply it to mask sites with the -9
            sentinel, which no (alt, ref) pair can express.  A zero-depth site
            is observed data and evidence for nullisomy, not a missing one.
        eps : float
            Per-read sequencing error rate.
        coverage : float
            Mean read depth at a disomic site.
        mismap_rate : float
            Background depth from mismapped reads, in copy-number units.
        r : float
            Intra-karyotype transition rate (recombination).
        a : float
            Inter-karyotype transition rate.
        unphased : bool
            Run the model in unphased mode.

        Returns
        -------
        alphas : np.ndarray
            Forward variable from HMM across k states.
        scaler : np.ndarray
            m-length array of scale parameters.
        states : list
            Tuple representation of states.
        karyotypes : np.ndarray
            Array of karyotypes in the MetaHMM model.
        loglik : float
            Total log-likelihood of the read counts.

        """
        self.check_read_inputs(
            alt_reads,
            ref_reads,
            pos,
            mat_haps,
            pat_haps,
            depths,
            eps,
            coverage,
            mismap_rate,
            r,
            a,
        )
        alphas, scaler, _, _, loglik = forward_algo_reads(
            alt_reads,
            ref_reads,
            pos,
            mat_haps,
            pat_haps,
            self.states,
            self.karyotypes,
            depths=depths,
            r=r,
            a=a,
            eps=eps,
            coverage=coverage,
            mismap_rate=mismap_rate,
            unphased=int(unphased),
        )
        return alphas, scaler, self.states, self.karyotypes, loglik

    def backward_algorithm(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        mat_haps,
        pat_haps,
        pi0=0.5,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
        mat_err=0.0,
        pat_err=0.0,
    ):
        """Backward HMM algorithm under a given statespace model.

        Parameters
        ----------
        bafs : np.ndarray
            B-allele frequencies across all m sites.
        lrrs : np.ndarray
            m-length array of log-R ratio values.
        sigmas : np.ndarray
            m-length array of per-site LRR noise standard deviations.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        pi0 : float
            Sparsity parameter for B-allele emission model.
        std_dev : float
            Standard deviation for B-allele emission model.
        r : float
            Intra-karyotype transition rate (recombination).
        a : float
            Inter-karyotype transition rate.
        unphased : bool
            Run the model in unphased mode.
        mat_err : float
            Per-site maternal genotyping error rate.
        pat_err : float
            Per-site paternal genotyping error rate.

        Returns
        -------
        betas : np.ndarray
            Backward variables from HMM across the k states.
        scaler : np.ndarray
            m-length array of scale parameters.
        states : list
            Tuple representation of states.
        karyotypes : np.ndarray
            Array of karyotypes in the MetaHMM model.
        loglik : float
            Total log-likelihood of B-allele frequency.

        """
        assert bafs.ndim == 1
        assert lrrs.ndim == 1
        assert sigmas.ndim == 1
        assert bafs.size == lrrs.size
        assert sigmas.size == bafs.size
        assert pos.ndim == 1
        assert (mat_haps.ndim == 2) & (pat_haps.ndim == 2)
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert bafs.size == mat_haps.shape[1]
        assert bafs.size == pos.size
        assert mat_haps.shape == pat_haps.shape
        assert np.all(pos[1:] > pos[:-1])
        assert r < 0.5 and r > 0
        assert a < 0.5 and a > 0
        assert 0.0 <= mat_err < 0.5
        assert 0.0 <= pat_err < 0.5
        betas, scaler, _, _, loglik = backward_algo(
            bafs,
            lrrs,
            sigmas,
            pos,
            mat_haps,
            pat_haps,
            self.states,
            self.karyotypes,
            r=r,
            a=a,
            pi0=pi0,
            std_dev=std_dev,
            unphased=int(unphased),
            mat_err=mat_err,
            pat_err=pat_err,
        )
        return betas, scaler, self.states, self.karyotypes, loglik

    def backward_algorithm_reads(
        self,
        alt_reads,
        ref_reads,
        pos,
        mat_haps,
        pat_haps,
        depths=None,
        eps=1e-2,
        coverage=1.0,
        mismap_rate=1e-2,
        r=1e-8,
        a=1e-2,
        unphased=False,
    ):
        """Backward HMM algorithm for read-level (sequencing) data.

        Mirror of :meth:`forward_algorithm_reads` -- same arguments, same
        emission model -- so the two report the same total log-likelihood and
        their outputs combine into a valid posterior.

        Parameters
        ----------
        alt_reads : np.ndarray
            m-length array of alternate-allele read counts.
        ref_reads : np.ndarray
            m-length array of reference-allele read counts.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        depths : np.ndarray, optional
            m-length array of total read depths, defaulting to
            ``alt_reads + ref_reads``.  Supply it to mask sites with the -9
            sentinel, which no (alt, ref) pair can express.  A zero-depth site
            is observed data and evidence for nullisomy, not a missing one.
        eps : float
            Per-read sequencing error rate.
        coverage : float
            Mean read depth at a disomic site.
        mismap_rate : float
            Background depth from mismapped reads, in copy-number units.
        r : float
            Intra-karyotype transition rate (recombination).
        a : float
            Inter-karyotype transition rate.
        unphased : bool
            Run the model in unphased mode.

        Returns
        -------
        betas : np.ndarray
            Backward variable from HMM across k states.
        scaler : np.ndarray
            m-length array of scale parameters.
        states : list
            Tuple representation of states.
        karyotypes : np.ndarray
            Array of karyotypes in the MetaHMM model.
        loglik : float
            Total log-likelihood of the read counts.

        """
        self.check_read_inputs(
            alt_reads,
            ref_reads,
            pos,
            mat_haps,
            pat_haps,
            depths,
            eps,
            coverage,
            mismap_rate,
            r,
            a,
        )
        betas, scaler, _, _, loglik = backward_algo_reads(
            alt_reads,
            ref_reads,
            pos,
            mat_haps,
            pat_haps,
            self.states,
            self.karyotypes,
            depths=depths,
            r=r,
            a=a,
            eps=eps,
            coverage=coverage,
            mismap_rate=mismap_rate,
            unphased=int(unphased),
        )
        return betas, scaler, self.states, self.karyotypes, loglik

    def forward_backward(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        mat_haps,
        pat_haps,
        pi0=0.2,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
        mat_err=0.0,
        pat_err=0.0,
    ):
        """Run the forward-backward algorithm across all states.

        Parameters
        ----------
        bafs : np.ndarray
            B-allele frequencies across all m sites.
        lrrs : np.ndarray
            m-length array of log-R ratio values.
        sigmas : np.ndarray
            m-length array of per-site LRR noise standard deviations.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        pi0 : float
            Sparsity parameter for B-allele emission model.
        std_dev : float
            Standard deviation for B-allele emission model.
        r : float
            Intra-karyotype transition rate (recombination).
        a : float
            Inter-karyotype transition rate.
        unphased : bool
            Run the model in unphased mode.
        mat_err : float
            Per-site maternal genotyping error rate.
        pat_err : float
            Per-site paternal genotyping error rate.

        Returns
        -------
        gammas : np.ndarray
            Log posterior density of being in each of k hidden states.
        states : list
            Tuple representation of states.
        karyotypes : np.ndarray
            Array of karyotypes in the model.

        """
        self.warn_missing_lrr(lrrs)
        alphas, _, states, karyotypes, _ = self.forward_algorithm(
            bafs,
            lrrs,
            sigmas,
            pos,
            mat_haps,
            pat_haps,
            pi0=pi0,
            std_dev=std_dev,
            r=r,
            a=a,
            unphased=unphased,
            mat_err=mat_err,
            pat_err=pat_err,
        )
        betas, _, _, _, _ = self.backward_algorithm(
            bafs,
            lrrs,
            sigmas,
            pos,
            mat_haps,
            pat_haps,
            pi0=pi0,
            std_dev=std_dev,
            r=r,
            a=a,
            unphased=unphased,
            mat_err=mat_err,
            pat_err=pat_err,
        )
        gammas = (alphas + betas) - logsumexp_sp(alphas + betas, axis=0)
        return gammas, states, karyotypes

    def forward_backward_reads(
        self,
        alt_reads,
        ref_reads,
        pos,
        mat_haps,
        pat_haps,
        depths=None,
        eps=1e-2,
        coverage=1.0,
        mismap_rate=1e-2,
        r=1e-8,
        a=1e-2,
        unphased=False,
    ):
        """Run the forward-backward algorithm on read-level (sequencing) data.

        Read-based counterpart of :meth:`forward_backward`.  The smoothed
        posterior it returns is the statistic to use for read data: the
        forward variable alone is filtered, and the read emission is sharp
        enough that its last column can sit confidently on the wrong state
        when the assumed recombination rate is misspecified.

        Parameters
        ----------
        alt_reads : np.ndarray
            m-length array of alternate-allele read counts.
        ref_reads : np.ndarray
            m-length array of reference-allele read counts.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        depths : np.ndarray, optional
            m-length array of total read depths, defaulting to
            ``alt_reads + ref_reads``.  Supply it to mask sites with the -9
            sentinel, which no (alt, ref) pair can express.  A zero-depth site
            is observed data and evidence for nullisomy, not a missing one.
        eps : float
            Per-read sequencing error rate.
        coverage : float
            Mean read depth at a disomic site.
        mismap_rate : float
            Background depth from mismapped reads, in copy-number units.
        r : float
            Intra-karyotype transition rate (recombination).
        a : float
            Inter-karyotype transition rate.
        unphased : bool
            Run the model in unphased mode.

        Returns
        -------
        gammas : np.ndarray
            Log posterior density of being in each of k hidden states.
        states : list
            Tuple representation of states.
        karyotypes : np.ndarray
            Array of karyotypes in the model.

        """
        kwargs = {
            "alt_reads": alt_reads,
            "ref_reads": ref_reads,
            "pos": pos,
            "mat_haps": mat_haps,
            "pat_haps": pat_haps,
            "depths": depths,
            "eps": eps,
            "coverage": coverage,
            "mismap_rate": mismap_rate,
            "r": r,
            "a": a,
            "unphased": unphased,
        }
        alphas, _, states, karyotypes, _ = self.forward_algorithm_reads(**kwargs)
        betas, _, _, _, _ = self.backward_algorithm_reads(**kwargs)
        gammas = (alphas + betas) - logsumexp_sp(alphas + betas, axis=0)
        return gammas, states, karyotypes

    def viterbi_algorithm(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        mat_haps,
        pat_haps,
        pi0=0.2,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
        mat_err=0.0,
        pat_err=0.0,
    ):
        """Implement the Viterbi traceback through karyotypic states.

        Parameters
        ----------
        bafs : np.ndarray
            B-allele frequencies across all m sites.
        lrrs : np.ndarray
            m-length array of log-R ratio values.
        sigmas : np.ndarray
            m-length array of per-site LRR noise standard deviations.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        pi0 : float
            Sparsity parameter for B-allele emission model.
        std_dev : float
            Standard deviation for B-allele emission model.
        r : float
            Intra-karyotype transition rate.
        a : float
            Inter-karyotype transition rate.
        unphased : bool
            Run the model in unphased mode.
        mat_err : float
            Per-site maternal genotyping error rate.
        pat_err : float
            Per-site paternal genotyping error rate.

        Returns
        -------
        path : np.ndarray
            Most likely copying path through k states in the model.
        states : list
            Tuple representation of states.
        deltas : np.ndarray
            Delta variable (maximum path probability at step m).
        psi : np.ndarray
            Storage vector for psi variable.

        """
        self.warn_missing_lrr(lrrs)
        assert bafs.ndim == 1
        assert lrrs.ndim == 1
        assert sigmas.ndim == 1
        assert bafs.size == lrrs.size
        assert lrrs.size == sigmas.size
        assert pos.ndim == 1
        assert (mat_haps.ndim == 2) & (pat_haps.ndim == 2)
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert bafs.size == pos.size
        assert bafs.size == mat_haps.shape[1]
        assert mat_haps.shape == pat_haps.shape
        assert np.all(pos[1:] > pos[:-1])
        assert 0.0 <= mat_err < 0.5
        assert 0.0 <= pat_err < 0.5
        path, states, deltas, psi = viterbi_algo(
            bafs,
            lrrs,
            sigmas,
            pos,
            mat_haps,
            pat_haps,
            self.states,
            self.karyotypes,
            r=r,
            a=a,
            pi0=pi0,
            std_dev=std_dev,
            unphased=int(unphased),
            mat_err=mat_err,
            pat_err=pat_err,
        )
        return path, states, deltas, psi

    def viterbi_algorithm_reads(
        self,
        alt_reads,
        ref_reads,
        pos,
        mat_haps,
        pat_haps,
        depths=None,
        eps=1e-2,
        coverage=1.0,
        mismap_rate=1e-2,
        r=1e-8,
        a=1e-2,
        unphased=False,
    ):
        """Viterbi traceback through karyotypic states on read-level data.

        Read-based counterpart of :meth:`viterbi_algorithm`.  Where
        :meth:`forward_backward_reads` gives the per-site marginal posterior,
        this gives the single most likely state path, which stays internally
        consistent across sites.

        Parameters
        ----------
        alt_reads : np.ndarray
            m-length array of alternate-allele read counts.
        ref_reads : np.ndarray
            m-length array of reference-allele read counts.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        depths : np.ndarray, optional
            m-length array of total read depths, defaulting to
            ``alt_reads + ref_reads``.  Supply it to mask sites with the -9
            sentinel, which no (alt, ref) pair can express.  A zero-depth site
            is observed data and evidence for nullisomy, not a missing one.
        eps : float
            Per-read sequencing error rate.
        coverage : float
            Mean read depth at a disomic site.
        mismap_rate : float
            Background depth from mismapped reads, in copy-number units.
        r : float
            Intra-karyotype transition rate (recombination).
        a : float
            Inter-karyotype transition rate.
        unphased : bool
            Run the model in unphased mode.

        Returns
        -------
        path : np.ndarray
            Most likely copying path through k states in the model.
        states : list
            Tuple representation of states.
        deltas : np.ndarray
            Delta variable (maximum path probability at step m).
        psi : np.ndarray
            Storage vector for psi variable.

        """
        self.check_read_inputs(
            alt_reads,
            ref_reads,
            pos,
            mat_haps,
            pat_haps,
            depths,
            eps,
            coverage,
            mismap_rate,
            r,
            a,
        )
        path, states, deltas, psi = viterbi_algo_reads(
            alt_reads,
            ref_reads,
            pos,
            mat_haps,
            pat_haps,
            self.states,
            self.karyotypes,
            depths=depths,
            r=r,
            a=a,
            eps=eps,
            coverage=coverage,
            mismap_rate=mismap_rate,
            unphased=int(unphased),
        )
        return path, states, deltas, psi

    def marginal_posterior_karyotypes(self, gammas, karyotypes):
        """Obtain the marginal posterior (not logged) probability over karyotypic states.

        Parameters
        ----------
        gammas : np.ndarray
            A k x m array of log-posterior density across sites.
        karyotypes : np.ndarray
            Karyotype labels for all k states.

        Returns
        -------
        gamma_karyo : np.ndarray
            Collapsed posteriors summed per unique karyotype.
        uniq_karyo : np.ndarray
            Unique karyotype labels in the order they appear.

        """
        assert gammas.ndim == 2
        assert gammas.shape[0] == karyotypes.size
        k, m = gammas.shape
        nk = np.unique(karyotypes).size
        gamma_karyo = np.zeros(shape=(nk, m))
        _, idx = np.unique(karyotypes, return_index=True)
        uniq_karyo = karyotypes[np.sort(idx)]
        for i, k in enumerate(uniq_karyo):
            # This is just the summed version of the posteriors ...
            gamma_karyo[i, :] = np.sum(np.exp(gammas[(karyotypes == k), :]), axis=0)
        return gamma_karyo, uniq_karyo

    def posterior_karyotypes(self, gammas, karyotypes):
        """Obtain full posterior on karyotypes chromosome-wide.

        This is the weighted proportion of time spent in each karyotypic state.

        Parameters
        ----------
        gammas : np.ndarray
            A k x m array of log-posterior density across sites.
        karyotypes : np.ndarray
            Karyotype labels for all k states.

        Returns
        -------
        kar_prob : dict
            Dictionary mapping karyotype label to chromosome-wide posterior probability.

        """
        assert gammas.ndim == 2
        assert gammas.shape[0] == karyotypes.size
        k, m = gammas.shape
        kar_prob = {}
        for k in np.unique(karyotypes):
            kar_prob[k] = np.sum(np.exp(gammas[(karyotypes == k), :])) / m
        return kar_prob

    def genotype_embryo(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        mat_haps,
        pat_haps,
        pi0=0.2,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
        viterbi=False,
    ):
        """Obtain genotype dosages for a putative disomic embryo.

        Parameters
        ----------
        bafs : np.ndarray
            B-allele frequencies across all m sites.
        lrrs : np.ndarray
            m-length array of log-R ratio values.
        sigmas : np.ndarray
            m-length array of per-site LRR noise standard deviations.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        pi0 : float
            Sparsity parameter for B-allele emission model.
        std_dev : float
            Standard deviation for B-allele emission model.
        r : float
            Intra-karyotype transition rate.
        a : float
            Inter-karyotype transition rate.
        unphased : bool
            Run the model in unphased mode.
        viterbi : bool
            Estimate the embryo genotypes using the Viterbi algorithm.

        Returns
        -------
        dosages : np.ndarray
            A 3 x m array of genotype probabilities (RR, RA, AA).

        """
        if self.aploid != "disomy":
            raise ValueError(
                "Obtaining non-disomic embryo genotypes is not currently supported!"
            )
        # Run the forward-backward algorithm on this ...
        if viterbi:
            path, states, _, _ = self.viterbi_algorithm(
                bafs=bafs,
                lrrs=lrrs,
                sigmas=sigmas,
                pos=pos,
                mat_haps=mat_haps,
                pat_haps=pat_haps,
                pi0=pi0,
                std_dev=std_dev,
                r=r,
                a=a,
                unphased=False,
            )
            dosages = np.zeros(shape=(3, pos.size), dtype=np.float32)
            for i, x in enumerate(path):
                cur_geno = int(
                    mat_dosage(mat_haps[:, i], states[x])
                    + pat_dosage(pat_haps[:, i], states[x])
                )
                assert (cur_geno >= 0) and (cur_geno <= 2)
                dosages[cur_geno, i] = 1.0
        else:
            gammas, states, _ = self.forward_backward(
                bafs=bafs,
                pos=pos,
                lrrs=lrrs,
                sigmas=sigmas,
                mat_haps=mat_haps,
                pat_haps=pat_haps,
                pi0=pi0,
                std_dev=std_dev,
                r=r,
                a=a,
                unphased=False,
            )
            # Calculate the genotype dosage of the alternative allele
            dosages = np.zeros(shape=(3, pos.size), dtype=np.float32)
            for i in range(pos.size):
                for j, x in enumerate(states):
                    cur_geno = int(
                        mat_dosage(mat_haps[:, i], x) + pat_dosage(pat_haps[:, i], x)
                    )
                    assert (cur_geno >= 0) and (cur_geno <= 2)
                    # This is analogous to the PL field (but on raw scale)
                    dosages[cur_geno, i] += np.exp(gammas[j, i])
            # Dosages are oriented towards the alt allele (bottom row is the alt/alt homozygote)
        return dosages

    def flag_parental_genotype_errors(
        self,
        gammas,
        states,
        bafs,
        mat_haps,
        pat_haps,
        pi0=0.2,
        std_dev=0.25,
    ):
        """Identify per-site parental genotype calls that are inconsistent with the posterior.

        For each site, computes a posterior-weighted log Bayes factor comparing the
        best alternative parental genotype (fixing the other parent) to the called one.
        A positive score means the data, given the inferred copying state, prefers a
        different parental genotype at that site.

        Parameters
        ----------
        gammas : np.ndarray
            k x m log-posterior array from forward_backward.
        states : list
            List of state tuples (output of forward_backward).
        bafs : np.ndarray
            m-length array of B-allele frequencies.
        mat_haps : np.ndarray
            2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            2 x m array of 0/1 paternal haplotypes.
        pi0 : float
            Sparsity parameter for BAF emission.
        std_dev : float
            Noise parameter for BAF emission.

        Returns
        -------
        mat_err_score : np.ndarray
            m-length array; positive values flag likely maternal genotype errors.
        pat_err_score : np.ndarray
            m-length array; positive values flag likely paternal genotype errors.

        """
        assert gammas.ndim == 2
        assert gammas.shape == (len(states), bafs.size)
        assert mat_haps.shape == (2, bafs.size)
        assert pat_haps.shape == (2, bafs.size)

        m = bafs.size
        mat_err_score = np.zeros(m)
        pat_err_score = np.zeros(m)
        all_genos = [[0, 0], [0, 1], [1, 0], [1, 1]]

        for i in range(m):
            mat_h = [int(mat_haps[0, i]), int(mat_haps[1, i])]
            pat_h = [int(pat_haps[0, i]), int(pat_haps[1, i])]
            for j, s in enumerate(states):
                w = np.exp(gammas[j, i])
                if w < 1e-12:
                    continue
                k = sum(v >= 0 for v in s)
                if k == 0:
                    continue
                m_d = mat_dosage(mat_h, s)
                p_d = pat_dosage(pat_h, s)
                called_loglik = emission_baf(
                    bafs[i], m_d, p_d, pi0=pi0, std_dev=std_dev, k=k
                )
                best_alt_mat = called_loglik
                for alt_mat in all_genos:
                    if alt_mat != mat_h:
                        alt_loglik = emission_baf(
                            bafs[i],
                            mat_dosage(alt_mat, s),
                            p_d,
                            pi0=pi0,
                            std_dev=std_dev,
                            k=k,
                        )
                        if alt_loglik > best_alt_mat:
                            best_alt_mat = alt_loglik
                mat_err_score[i] += w * (best_alt_mat - called_loglik)
                best_alt_pat = called_loglik
                for alt_pat in all_genos:
                    if alt_pat != pat_h:
                        alt_loglik = emission_baf(
                            bafs[i],
                            m_d,
                            pat_dosage(alt_pat, s),
                            pi0=pi0,
                            std_dev=std_dev,
                            k=k,
                        )
                        if alt_loglik > best_alt_pat:
                            best_alt_pat = alt_loglik
                pat_err_score[i] += w * (best_alt_pat - called_loglik)

        return mat_err_score, pat_err_score


class QuadHMM(AneuploidyHMM):
    """Joint HMM for two sibling embryos used to detect inter-sibling haplotype sharing.

    Models the joint copying state of two euploid siblings as a pair of independent
    disomy states drawn from the same parental haplotypes.  The 16-state product
    space (4 × 4 single-embryo disomy states) is collapsed into four Roach et al.
    2010 identity-by-descent classes:

    - 0: maternally haploidentical (siblings share the same maternal haplotype)
    - 1: paternally haploidentical (siblings share the same paternal haplotype)
    - 2: fully identical (same maternal *and* paternal haplotype)
    - 3: non-identical (different maternal and paternal haplotypes)

    Transitions between copying states are governed by a single recombination rate
    parameter ``r``; the BAF emission for each sibling follows the same Gaussian
    mixture model as :class:`MetaHMM` but with per-sibling ``pi0`` and ``std_dev``.

    Attributes
    ----------
    states : list of tuple of tuple
        16 joint copying states, each ``((m0, _, p0, _), (m1, _, p1, _))``.
    karyotypes : np.ndarray of str
        All entries are ``"2"`` (disomy assumed for both siblings).
    """

    def __init__(self):
        """Initialize the QuadHMM model."""
        self.ploidy = 2
        self.aploid = "2"
        self.single_states = [
            (0, -1, 0, -1),
            (0, -1, 1, -1),
            (1, -1, 0, -1),
            (1, -1, 1, -1),
        ]
        self.states = []
        for i in self.single_states:
            for j in self.single_states:
                self.states.append((i, j))
        self.karyotypes = np.repeat("2", len(self.states)).astype(str)

    def forward_algorithm(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        pi0=(0.7, 0.7),
        std_dev=(0.15, 0.15),
        r=1e-8,
        mat_err=0.0,
        pat_err=0.0,
    ):
        """Implement the forward algorithm for QuadHMM model.

        Parameters
        ----------
        bafs : list
            List of two np.ndarray arrays of B-allele frequencies across m sites for two siblings.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        pi0 : tuple of float
            Sparsity parameters for B-allele emission model, one per sibling.
        std_dev : tuple of float
            Standard deviation parameters for B-allele emission model, one per sibling.
        r : float
            Inter-state transition rate.
        mat_err : float
            Per-site maternal genotyping error rate.
        pat_err : float
            Per-site paternal genotyping error rate.

        Returns
        -------
        alphas : np.ndarray
            Forward variable from HMM across the joint sibling states.
        scaler : np.ndarray
            m-length array of scale parameters.
        states : list
            Tuple representation of the 16 joint states.
        karyotypes : np.ndarray
            Array of karyotypes (all entries ``"2"``).
        loglik : float
            Total log-likelihood of joint sibling B-allele frequencies.

        """
        assert len(bafs) == 2
        assert bafs[0].size == pos.size
        assert bafs[0].size == bafs[1].size
        assert bafs[0].size == mat_haps.shape[1]
        assert (mat_haps.ndim == 2) and (pat_haps.ndim == 2)
        assert mat_haps.size == pat_haps.size
        assert np.all(pos[1:] > pos[:-1])
        alphas, scaler, states, karyotypes, loglik = forward_algo_sibs(
            bafs,
            pos,
            mat_haps,
            pat_haps,
            states=self.states,
            karyotypes=self.karyotypes,
            r=r,
            pi0=pi0,
            std_dev=std_dev,
            mat_err=mat_err,
            pat_err=pat_err,
        )
        return alphas, scaler, states, karyotypes, loglik

    def backward_algorithm(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        pi0=(0.7, 0.7),
        std_dev=(0.15, 0.15),
        r=1e-8,
        mat_err=0.0,
        pat_err=0.0,
    ):
        """Implement the backward algorithm for QuadHMM model.

        Parameters
        ----------
        bafs : list
            List of two np.ndarray arrays of B-allele frequencies across m sites for two siblings.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        pi0 : tuple of float
            Sparsity parameters for B-allele emission model, one per sibling.
        std_dev : tuple of float
            Standard deviation parameters for B-allele emission model, one per sibling.
        r : float
            Inter-state transition rate.
        mat_err : float
            Per-site maternal genotyping error rate.
        pat_err : float
            Per-site paternal genotyping error rate.

        Returns
        -------
        betas : np.ndarray
            Backward variable from HMM across the joint sibling states.
        scaler : np.ndarray
            m-length array of scale parameters.
        states : list
            Tuple representation of the 16 joint states.
        karyotypes : np.ndarray
            Array of karyotypes (all entries ``"2"``).
        loglik : float
            Total log-likelihood of joint sibling B-allele frequencies.

        """
        assert len(bafs) == 2
        assert bafs[0].size == pos.size
        assert bafs[0].size == bafs[1].size
        assert bafs[0].size == mat_haps.shape[1]
        assert (mat_haps.ndim == 2) and (pat_haps.ndim == 2)
        assert mat_haps.size == pat_haps.size
        assert np.all(pos[1:] > pos[:-1])
        alphas, scaler, states, karyotypes, loglik = backward_algo_sibs(
            bafs,
            pos,
            mat_haps,
            pat_haps,
            states=self.states,
            karyotypes=self.karyotypes,
            r=r,
            pi0=pi0,
            std_dev=std_dev,
            mat_err=mat_err,
            pat_err=pat_err,
        )
        return alphas, scaler, states, karyotypes, loglik

    def forward_backward(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        pi0=(0.7, 0.7),
        std_dev=(0.15, 0.15),
        r=1e-8,
        a=1e-2,
        mat_err=0.0,
        pat_err=0.0,
    ):
        """Implement the forward-backward algorithm for the QuadHMM model.

        Parameters
        ----------
        bafs : list
            List of two np.ndarray arrays of B-allele frequencies across m sites for two siblings.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        pi0 : tuple of float
            Sparsity parameters for B-allele emission model, one per sibling.
        std_dev : tuple of float
            Standard deviation parameters for B-allele emission model, one per sibling.
        r : float
            Inter-state transition rate.
        a : float
            Unused; retained for API compatibility.
        mat_err : float
            Per-site maternal genotyping error rate.
        pat_err : float
            Per-site paternal genotyping error rate.

        Returns
        -------
        gammas : np.ndarray
            Log posterior density of being in each of the joint sibling states.
        states : list
            Tuple representation of the 16 joint states.
        karyotypes : None
            Always returns None for QuadHMM.

        """
        alphas, _, states, _, _ = forward_algo_sibs(
            bafs,
            pos,
            mat_haps,
            pat_haps,
            states=self.states,
            karyotypes=self.karyotypes,
            pi0=pi0,
            std_dev=std_dev,
            r=r,
            mat_err=mat_err,
            pat_err=pat_err,
        )
        betas, _, _, _, _ = backward_algo_sibs(
            bafs,
            pos,
            mat_haps,
            pat_haps,
            states=self.states,
            karyotypes=self.karyotypes,
            pi0=pi0,
            std_dev=std_dev,
            r=r,
            mat_err=mat_err,
            pat_err=pat_err,
        )
        gammas = (alphas + betas) - logsumexp_sp(alphas + betas, axis=0)
        return gammas, states, None

    def viterbi_algorithm(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        pi0=(0.7, 0.7),
        std_dev=(0.15, 0.15),
        r=1e-8,
        mat_err=0.0,
        pat_err=0.0,
    ):
        """Viterbi algorithm for the QuadHMM model.

        Parameters
        ----------
        bafs : list
            List of two np.ndarray arrays of B-allele frequencies across m sites for two siblings.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        pi0 : tuple of float
            Sparsity parameters for B-allele emission model, one per sibling.
        std_dev : tuple of float
            Standard deviation parameters for B-allele emission model, one per sibling.
        r : float
            Inter-state transition rate.
        mat_err : float
            Per-site maternal genotyping error rate.
        pat_err : float
            Per-site paternal genotyping error rate.

        Returns
        -------
        path : np.ndarray
            Most likely copying path through the joint sibling states.
        states : list
            Tuple representation of the 16 joint states.
        deltas : np.ndarray
            Delta variable (maximum path probability at each site).
        psi : np.ndarray
            Storage vector for psi variable.

        """
        path, states, deltas, psi = viterbi_algo_sibs(
            bafs,
            pos,
            mat_haps,
            pat_haps,
            states=self.states,
            karyotypes=self.karyotypes,
            pi0=pi0,
            std_dev=std_dev,
            r=r,
            mat_err=mat_err,
            pat_err=pat_err,
        )
        return path, states, deltas, psi

    def restrict_path(self, path):
        """Break down states into the same categories as Roach et al for determining recombinations.

        The state definitions used for tracing sibling haplotypes are:

        - 0: maternal-haploidentical
        - 1: paternal-haploidentical
        - 2: identical
        - 3: non-identical

        Parameters
        ----------
        path : np.ndarray
            A sequence of states output from the Viterbi algorithm (16 possible states).

        Returns
        -------
        refined_path : np.ndarray
            A sequence of m states in 0, 1, 2, 3 indicating states from Roach et al.

        """
        maternal_haploidentical = []
        paternal_haploidentical = []
        identical = []
        non_identical = []
        for i, (x, y) in enumerate(self.states):
            if x == y:
                identical.append(i)
            elif (x[0] == y[0]) and (x[2] != y[2]):
                maternal_haploidentical.append(i)
            elif (x[2] == y[2]) and (x[0] != y[0]):
                paternal_haploidentical.append(i)
            else:
                non_identical.append(i)
        # Refining the path estimation to only the Roach et al 2010 states
        refined_path = np.zeros(path.size)
        for i in range(path.size):
            if path[i] in maternal_haploidentical:
                refined_path[i] = 0
            elif path[i] in paternal_haploidentical:
                refined_path[i] = 1
            elif path[i] in identical:
                refined_path[i] = 2
            elif path[i] in non_identical:
                refined_path[i] = 3
            else:
                raise ValueError("Incorrect path estimate!")
        return refined_path

    def restrict_states(self):
        """Break down states into the same categories as Roach et al for determining recombinations.

        Returns
        -------
        maternal_haploidentical : list
            Indexes of maternally haploidentical states.
        paternal_haploidentical : list
            Indexes of paternally haploidentical states.
        identical : list
            Indexes of identical states (siblings share same haplotypes).
        non_identical : list
            Indexes of non-identical states.

        """
        maternal_haploidentical = []
        paternal_haploidentical = []
        identical = []
        non_identical = []
        for i, (x, y) in enumerate(self.states):
            if x == y:
                identical.append(i)
            elif (x[0] == y[0]) and (x[2] != y[2]):
                maternal_haploidentical.append(i)
            elif (x[2] == y[2]) and (x[0] != y[0]):
                paternal_haploidentical.append(i)
            else:
                non_identical.append(i)
        return (
            maternal_haploidentical,
            paternal_haploidentical,
            identical,
            non_identical,
        )

    def viterbi_path(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        pi0=(0.7, 0.7),
        std_dev=(0.15, 0.15),
        r=1e-8,
    ):
        """Obtain the restricted Viterbi path for traceback.

        Parameters
        ----------
        bafs : list
            List of two np.ndarray arrays of B-allele frequencies across m sites for two siblings.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        pi0 : tuple of float
            Sparsity parameters for B-allele emission model, one per sibling.
        std_dev : tuple of float
            Standard deviation parameters for B-allele emission model, one per sibling.
        r : float
            Inter-state transition rate.

        Returns
        -------
        res_path : np.ndarray
            Most likely copying path restricted to 4 states from Roach et al.

        """
        path, _, _, _ = self.viterbi_algorithm(
            bafs, pos, mat_haps, pat_haps, pi0=pi0, std_dev=std_dev, r=r
        )
        res_path = self.restrict_path(path)
        return res_path

    def map_path(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        pi0=(0.7, 0.7),
        std_dev=(0.15, 0.15),
        r=1e-8,
    ):
        """Obtain the Maximum A-Posteriori Path across restricted states.

        Parameters
        ----------
        bafs : list
            List of two np.ndarray arrays of B-allele frequencies across m sites for two siblings.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        pi0 : tuple of float
            Sparsity parameters for B-allele emission model, one per sibling.
        std_dev : tuple of float
            Standard deviation parameters for B-allele emission model, one per sibling.
        r : float
            Inter-state transition rate.

        Returns
        -------
        map_path : np.ndarray
            Max-a-posteriori copying path through 4 sibling copying states.

        """
        gammas, _, _ = self.forward_backward(
            bafs, pos, mat_haps, pat_haps, pi0=pi0, std_dev=std_dev, r=r
        )
        (
            maternal_haploidentical,
            paternal_haploidentical,
            identical,
            non_identical,
        ) = self.restrict_states()
        red_gammas = np.zeros(shape=(4, gammas.shape[1]))
        red_gammas[0, :] = np.exp(gammas)[maternal_haploidentical, :].sum(axis=0)
        red_gammas[1, :] = np.exp(gammas)[paternal_haploidentical, :].sum(axis=0)
        red_gammas[2, :] = np.exp(gammas)[identical, :].sum(axis=0)
        red_gammas[3, :] = np.exp(gammas)[non_identical, :].sum(axis=0)
        red_gammas = np.log(red_gammas)
        map_path = np.argmax(red_gammas, axis=0)
        return map_path

    def flag_parental_genotype_errors(
        self,
        gammas,
        states,
        bafs,
        mat_haps,
        pat_haps,
        pi0=(0.7, 0.7),
        std_dev=(0.15, 0.15),
    ):
        """Identify per-site parental genotype calls that are inconsistent with the posterior.

        For each site, computes a posterior-weighted log Bayes factor comparing the
        best alternative parental genotype (fixing the other parent) to the called one.
        Both siblings observe the same parental genotype, so the joint sibling BAF
        likelihood is used when scoring alternatives.  A positive score means the
        data, given the inferred copying state, prefers a different parental genotype
        at that site.

        Parameters
        ----------
        gammas : np.ndarray
            k x m log-posterior array from forward_backward.
        states : list
            List of paired state tuples (output of forward_backward).
        bafs : list
            List of two m-length np.ndarray BAF arrays for the two siblings.
        mat_haps : np.ndarray
            2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            2 x m array of 0/1 paternal haplotypes.
        pi0 : tuple
            (pi0_sib0, pi0_sib1) sparsity parameters for BAF emission.
        std_dev : tuple
            (std_sib0, std_sib1) noise parameters for BAF emission.

        Returns
        -------
        mat_err_score : np.ndarray
            m-length array; positive values flag likely maternal genotype errors.
        pat_err_score : np.ndarray
            m-length array; positive values flag likely paternal genotype errors.

        """
        assert gammas.ndim == 2
        assert len(bafs) == 2
        assert bafs[0].size == bafs[1].size
        assert gammas.shape == (len(states), bafs[0].size)
        assert mat_haps.shape == (2, bafs[0].size)
        assert pat_haps.shape == (2, bafs[0].size)

        m = bafs[0].size
        mat_err_score = np.zeros(m)
        pat_err_score = np.zeros(m)
        all_genos = [[0, 0], [0, 1], [1, 0], [1, 1]]

        for i in range(m):
            mat_h = [int(mat_haps[0, i]), int(mat_haps[1, i])]
            pat_h = [int(pat_haps[0, i]), int(pat_haps[1, i])]
            for j, (s0, s1) in enumerate(states):
                w = np.exp(gammas[j, i])
                if w < 1e-12:
                    continue
                m_d0 = mat_dosage(mat_h, s0)
                p_d0 = pat_dosage(pat_h, s0)
                m_d1 = mat_dosage(mat_h, s1)
                p_d1 = pat_dosage(pat_h, s1)
                called_loglik = emission_baf(
                    bafs[0][i], m_d0, p_d0, pi0=pi0[0], std_dev=std_dev[0], k=2
                ) + emission_baf(
                    bafs[1][i], m_d1, p_d1, pi0=pi0[1], std_dev=std_dev[1], k=2
                )
                best_alt_mat = called_loglik
                for alt_mat in all_genos:
                    if alt_mat != mat_h:
                        alt_loglik = emission_baf(
                            bafs[0][i],
                            mat_dosage(alt_mat, s0),
                            p_d0,
                            pi0=pi0[0],
                            std_dev=std_dev[0],
                            k=2,
                        ) + emission_baf(
                            bafs[1][i],
                            mat_dosage(alt_mat, s1),
                            p_d1,
                            pi0=pi0[1],
                            std_dev=std_dev[1],
                            k=2,
                        )
                        if alt_loglik > best_alt_mat:
                            best_alt_mat = alt_loglik
                mat_err_score[i] += w * (best_alt_mat - called_loglik)
                best_alt_pat = called_loglik
                for alt_pat in all_genos:
                    if alt_pat != pat_h:
                        alt_loglik = emission_baf(
                            bafs[0][i],
                            m_d0,
                            pat_dosage(alt_pat, s0),
                            pi0=pi0[0],
                            std_dev=std_dev[0],
                            k=2,
                        ) + emission_baf(
                            bafs[1][i],
                            m_d1,
                            pat_dosage(alt_pat, s1),
                            pi0=pi0[1],
                            std_dev=std_dev[1],
                            k=2,
                        )
                        if alt_loglik > best_alt_pat:
                            best_alt_pat = alt_loglik
                pat_err_score[i] += w * (best_alt_pat - called_loglik)

        return mat_err_score, pat_err_score

    def det_recomb_sex(self, i, j):
        """Determine the parental origin of the recombination event.

        Parameters
        ----------
        i : int
            State index for the previous state.
        j : int
            State index for the current state.

        Returns
        -------
        m : int
            Sex of haplotype on which the transition occurred (0: maternal, 1: paternal).

        """
        assert i != j
        m = -1
        if i == 0 and j == 3:
            # maternal haploidentity -> non-identity
            m = 0
        if i == 0 and j == 2:
            # maternal haploidentity -> identity
            m = 1
        if i == 1 and j == 3:
            # paternal haploidentity -> non-identity
            m = 1
        if i == 1 and j == 2:
            # paternal haploidentity -> identity
            m = 0
        if i == 2 and j == 0:
            # identical -> maternal haploidentity
            m = 1
        if i == 2 and j == 1:
            # identical -> paternal haploidentity
            m = 0
        if i == 3 and j == 0:
            # non-identical -> maternal haploidentity
            m = 0
        if i == 3 and j == 1:
            # non-identical -> paternal haploidentity
            m = 1
        return m

    def isolate_recomb(self, path_xy, path_xzs, window=20):
        """Isolate key recombination events from a pair of refined paths.

        Parameters
        ----------
        path_xy : np.ndarray
            Array of path indices for the focal pair of siblings.
        path_xzs : list
            List of sibling paths (each a np.ndarray).
        window : int
            Number of SNPs within which the closest transition must fall (minimum resolution).

        Returns
        -------
        mat_recomb_lst : list
            List of maternal recombination SNP indexes.
        pat_recomb_lst : list
            List of paternal recombination SNP indexes.
        mat_recomb : dict
            Dictionary of SNP-index to sibling-count for maternal recombinations.
        pat_recomb : dict
            Dictionary of SNP-index to sibling-count for paternal recombinations.

        """
        mat_recomb = {}
        pat_recomb = {}
        for path_xz in path_xzs:
            assert path_xy.size == path_xz.size
            transitions_01 = np.where(path_xy[:-1] != path_xy[1:])[0]
            transitions_02 = np.where(path_xz[:-1] != path_xz[1:])[0]
            for r in transitions_01:
                # This is the targetted transition that we want to assign to maternal or paternal position.
                i0, j0 = path_xy[r], path_xy[r + 1]
                dists = np.sqrt((transitions_02 - r) ** 2)
                if np.any(dists < window):
                    # This gets the closest matching one
                    r2 = transitions_02[np.argmin(dists)]
                    i1, j1 = path_xz[r2], path_xz[r2 + 1]
                    m = self.det_recomb_sex(i0, j0)
                    m2 = self.det_recomb_sex(i1, j1)
                    if m == 0 and m2 == 0:
                        if r not in mat_recomb:
                            mat_recomb[r] = 1
                        else:
                            mat_recomb[r] = mat_recomb[r] + 1
                    if m == 1 and m2 == 1:
                        if r not in pat_recomb:
                            pat_recomb[r] = 1
                        else:
                            pat_recomb[r] = pat_recomb[r] + 1

        # NOTE: here we just get positions, and if they are supported by the majority rule ...
        mat_recomb_lst = [k for k in mat_recomb if mat_recomb[k] > len(path_xzs) / 2]
        pat_recomb_lst = [k for k in pat_recomb if pat_recomb[k] > len(path_xzs) / 2]
        # This returns the list of tuples on the recombination positions and minimum distances across the traces.
        return mat_recomb_lst, pat_recomb_lst, mat_recomb, pat_recomb

    def est_sharing(
        self,
        bafs,
        pos,
        mat_haps,
        pat_haps,
        pi0=(0.7, 0.7),
        std_dev=(0.15, 0.15),
        r=1e-8,
    ):
        """Estimate the proportion of the chromosome that is shared identically across siblings.

        Parameters
        ----------
        bafs : list
            List of two np.ndarray arrays of B-allele frequencies across m sites for two siblings.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        pi0 : tuple of float
            Sparsity parameters for B-allele emission model, one per sibling.
        std_dev : tuple of float
            Standard deviation parameters for B-allele emission model, one per sibling.
        r : float
            Inter-state transition rate.

        Returns
        -------
        mat_haplo_len : float
            Total base-pair length of maternally haploidentical regions.
        pat_haplo_len : float
            Total base-pair length of paternally haploidentical regions.
        both_haplo_len : float
            Total base-pair length of fully identical (both-haplo) regions.
        tot_len : float
            Total base-pair length of the typed region on the chromosome.

        """
        # 1. Estimate the simplified viterbi path through these states
        cur_viterbi_path = self.viterbi_path(
            bafs=bafs,
            pos=pos,
            mat_haps=mat_haps,
            pat_haps=pat_haps,
            pi0=pi0,
            std_dev=std_dev,
            r=r,
        )
        # 2. Estimate the amount of the chromosome in haplo-identical states
        maternal_haplo_idx = np.where(cur_viterbi_path == 0)[0]
        paternal_haplo_idx = np.where(cur_viterbi_path == 1)[0]
        both_haplo_idx = np.where(cur_viterbi_path == 2)[0]
        # 3. Now estimate the fraction of the genome explicitly ...
        aggregate_length = lambda p, idx: np.sum(
            [
                p1 - p0
                for (p0, p1, x0, x1) in zip(
                    p[idx][:-1],
                    p[idx][1:],
                    idx[:-1],
                    idx[1:],
                )
                if (x1 - 1 == x0)
            ]
        )
        tot_len = pos[-1] - pos[0]
        mat_haplo_len = aggregate_length(pos, maternal_haplo_idx)
        pat_haplo_len = aggregate_length(pos, paternal_haplo_idx)
        both_haplo_len = aggregate_length(pos, both_haplo_idx)
        return mat_haplo_len, pat_haplo_len, both_haplo_len, tot_len


class PocHMM(MetaHMM):
    """Aneuploidy HMM for products-of-conception with a single observed parent.

    Extends :class:`MetaHMM` to handle mother–child or father–child duos in which
    only one parent's haplotypes are available.  The unobserved parent's genotype is
    marginalised out at each site using population allele frequencies supplied as
    ``freqs``.  When ``freqs`` is ``None`` the forward and backward passes give
    the three unobserved-parent genotypes equal prior weight at every site.

    The forward and backward algorithms are re-implemented here via a dedicated
    Cython kernel (``forward_algo_duo`` / ``backward_algo_duo``) that handles the
    marginalisation efficiently.  The inherited Viterbi algorithm expects both
    parents' haplotypes and does not apply to duos.

    Parameters
    ----------
    disomy : bool, optional
        Restrict the state space to disomy-only, as in :class:`MetaHMM`.
        Default ``False``.
    upd : bool, optional
        Add uniparental disomy states to the state space.  As with
        :class:`MetaHMM`, LRR data helps distinguish UPD from normal disomy.
        In the duo setting, only the UPD states on the *observed* parent's
        side (see ``maternal`` in :meth:`forward_algorithm`) are identifiable
        from copy-number-neutral signal alone; UPD states on the unobserved
        side are marginalised over population allele frequencies like any
        other unobserved-parent state.  Default ``False``.
    """

    def __init__(self, disomy=False, upd=False):
        """Initialize the PocHMM with the full MetaHMM state space."""
        super().__init__(disomy=disomy, upd=upd)

    def est_sigma_pi0(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        haps,
        freqs=None,
        maternal=True,
        algo="L-BFGS-B",
        pi0_bounds=(0.01, 0.99),
        sigma_bounds=(1e-2, 0.5),
        **kwargs,
    ):
        """Estimate sigma and pi0 under the B-Allele Frequency model using optimization of forward algorithm likelihood.

        Parameters
        ----------
        bafs : np.ndarray
            B-allele frequencies across all m sites.
        lrrs : np.ndarray
            m-length array of log-R ratio values.
        sigmas : np.ndarray
            m-length array of per-site LRR noise standard deviations.
        pos : np.ndarray
            Basepair positions of the SNPs.
        haps : np.ndarray
            A 2 x m array of 0/1 parental haplotypes.
        freqs : np.ndarray or None
            An m-length array of allele frequencies as priors.
        maternal : bool
            If True, haps are maternal haplotypes; otherwise paternal.
        algo : str
            One of Nelder-Mead, L-BFGS-B, or Powell algorithms for optimization.
        pi0_bounds : tuple
            Bounds for acceptable values of pi0 parameter.
        sigma_bounds : tuple
            Bounds for acceptable values for sigma.
        **kwargs
            Passed to :meth:`forward_algorithm`.

        Returns
        -------
        pi0_est : float
            Estimate of sparsity parameter (pi0) for B-allele emission model.
        sigma_est : float
            Estimate of noise parameter (sigma) for B-allele emission model.

        """
        assert algo in ["Nelder-Mead", "L-BFGS-B", "Powell"]
        assert (len(pi0_bounds) == 2) and (len(sigma_bounds) == 2)
        assert (pi0_bounds[0] > 0) and (pi0_bounds[1] > 0)
        assert (pi0_bounds[0] < 1) and (pi0_bounds[1] < 1)
        assert pi0_bounds[0] < pi0_bounds[1]
        assert (sigma_bounds[0] > 0) and (sigma_bounds[1] > 0)
        assert sigma_bounds[0] < sigma_bounds[1]
        mid_pi0 = np.mean(pi0_bounds)
        mid_sigma = np.mean(sigma_bounds)
        opt_res = minimize(
            lambda x: (
                -self.forward_algorithm(
                    bafs=bafs,
                    lrrs=lrrs,
                    sigmas=sigmas,
                    pos=pos,
                    haps=haps,
                    freqs=freqs,
                    maternal=maternal,
                    pi0=x[0],
                    std_dev=x[1],
                    **kwargs,
                )[4]
            ),
            x0=[mid_pi0, mid_sigma],
            method=algo,
            bounds=[pi0_bounds, sigma_bounds],
            tol=1e-4,
            options={"disp": True, "ftol": 1e-4, "xtol": 1e-4},
        )
        pi0_est = opt_res.x[0]
        sigma_est = opt_res.x[1]
        return pi0_est, sigma_est

    def forward_algorithm(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        haps,
        freqs=None,
        maternal=True,
        pi0=0.2,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
        obs_err=0.0,
        mcc=0.0,
        delta=0.0,
    ):
        """Forward algorithm for duos.

        Parameters
        ----------
        bafs : np.ndarray
            B-allele frequencies across all m sites.
        lrrs : np.ndarray
            m-length array of log-R ratio values.
        sigmas : np.ndarray
            m-length array of per-site LRR noise standard deviations.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        haps : np.ndarray
            A 2 x m array of 0/1 parental haplotypes.
        freqs : np.ndarray or None
            An m-length array of population allele frequencies.
        maternal : bool
            If True, haps are maternal haplotypes; otherwise paternal.
        pi0 : float
            Sparsity parameter for B-allele emission model.
        std_dev : float
            Standard deviation for B-allele emission model.
        r : float
            Intra-karyotype transition rate (recombination).
        a : float
            Scale on ``r`` giving the inter-karyotype transition rate.
        unphased : bool
            Unused; kept for signature compatibility with :class:`MetaHMM`.
        obs_err : float
            Per-site genotyping error rate for the observed parent.
        mcc : float
            Fraction of maternal cells contaminating the sample.
        delta : float
            Sample-level LRR offset from array normalisation.

        Returns
        -------
        alphas : np.ndarray
            Forward variable from HMM across k states.
        scaler : np.ndarray
            m-length array of scale parameters.
        states : list
            Tuple representation of states.
        karyotypes : np.ndarray
            Array of karyotypes in the MetaHMM model.
        loglik : float
            Total log-likelihood of the BAF and LRR data.

        """
        assert bafs.ndim == 1
        assert lrrs.ndim == 1
        assert sigmas.ndim == 1
        assert bafs.size == lrrs.size
        assert lrrs.size == sigmas.size
        assert pos.ndim == 1
        assert haps.ndim == 2
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert bafs.size == haps.shape[1]
        assert bafs.size == pos.size
        assert np.all(pos[1:] > pos[:-1])
        assert r < 0.5 and r > 0
        assert a < 0.5 and a > 0
        assert 0.0 <= obs_err < 0.5
        assert 0.0 <= mcc < 1.0
        if freqs is not None:
            assert freqs.size == bafs.size
        else:
            freqs = np.repeat(-1, bafs.size)
        alphas, scaler, _, _, loglik = forward_algo_duo(
            bafs,
            lrrs,
            sigmas,
            pos,
            haps,
            freqs,
            self.states,
            self.karyotypes,
            maternal=maternal,
            r=r,
            a=a,
            pi0=pi0,
            std_dev=std_dev,
            obs_err=obs_err,
            mcc=mcc,
            delta=delta,
        )
        return alphas, scaler, self.states, self.karyotypes, loglik

    def backward_algorithm(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        haps,
        freqs=None,
        maternal=True,
        pi0=0.2,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
        obs_err=0.0,
        mcc=0.0,
        delta=0.0,
    ):
        """Backward algorithm for duos.

        Parameters
        ----------
        bafs : np.ndarray
            B-allele frequencies across all m sites.
        lrrs : np.ndarray
            m-length array of log-R ratio values.
        sigmas : np.ndarray
            m-length array of per-site LRR noise standard deviations.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        haps : np.ndarray
            A 2 x m array of 0/1 parental haplotypes.
        freqs : np.ndarray or None
            An m-length array of population allele frequencies.
        maternal : bool
            If True, haps are maternal haplotypes; otherwise paternal.
        pi0 : float
            Sparsity parameter for B-allele emission model.
        std_dev : float
            Standard deviation for B-allele emission model.
        r : float
            Intra-karyotype transition rate (recombination).
        a : float
            Scale on ``r`` giving the inter-karyotype transition rate.
        unphased : bool
            Unused; kept for signature compatibility with :class:`MetaHMM`.
        obs_err : float
            Per-site genotyping error rate for the observed parent.
        mcc : float
            Fraction of maternal cells contaminating the sample.
        delta : float
            Sample-level LRR offset from array normalisation.

        Returns
        -------
        betas : np.ndarray
            Backward variable from HMM across k states.
        scaler : np.ndarray
            m-length array of scale parameters.
        states : list
            Tuple representation of states.
        karyotypes : np.ndarray
            Array of karyotypes in the MetaHMM model.
        loglik : float
            Total log-likelihood of the BAF and LRR data.

        """
        assert bafs.ndim == 1
        assert lrrs.ndim == 1
        assert sigmas.ndim == 1
        assert bafs.size == lrrs.size
        assert lrrs.size == sigmas.size
        assert pos.ndim == 1
        assert haps.ndim == 2
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert bafs.size == haps.shape[1]
        assert bafs.size == pos.size
        assert np.all(pos[1:] > pos[:-1])
        assert r < 0.5 and r > 0
        assert a < 0.5 and a > 0
        assert 0.0 <= obs_err < 0.5
        assert 0.0 <= mcc < 1.0
        if freqs is not None:
            assert freqs.size == bafs.size
        else:
            freqs = np.repeat(-1, bafs.size)
        betas, scaler, _, _, loglik = backward_algo_duo(
            bafs,
            lrrs,
            sigmas,
            pos,
            haps,
            freqs,
            self.states,
            self.karyotypes,
            maternal=maternal,
            r=r,
            a=a,
            pi0=pi0,
            std_dev=std_dev,
            obs_err=obs_err,
            mcc=mcc,
            delta=delta,
        )
        return betas, scaler, self.states, self.karyotypes, loglik

    def forward_backward(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        haps,
        freqs=None,
        maternal=True,
        pi0=0.2,
        std_dev=0.25,
        r=1e-8,
        a=1e-2,
        unphased=False,
        obs_err=0.0,
        mcc=0.0,
        delta=0.0,
    ):
        """Forward-backward algorithm for duos.

        Parameters
        ----------
        bafs : np.ndarray
            B-allele frequencies across all m sites.
        lrrs : np.ndarray
            m-length array of log-R ratio values.
        sigmas : np.ndarray
            m-length array of per-site LRR noise standard deviations.
        pos : np.ndarray
            m-length vector of basepair positions for sites.
        haps : np.ndarray
            A 2 x m array of 0/1 parental haplotypes.
        freqs : np.ndarray or None
            An m-length array of population allele frequencies.
        maternal : bool
            If True, haps are maternal haplotypes; otherwise paternal.
        pi0 : float
            Sparsity parameter for B-allele emission model.
        std_dev : float
            Standard deviation for B-allele emission model.
        r : float
            Intra-karyotype transition rate (recombination).
        a : float
            Scale on ``r`` giving the inter-karyotype transition rate.
        unphased : bool
            Unused; kept for signature compatibility with :class:`MetaHMM`.
        obs_err : float
            Per-site genotyping error rate for the observed parent.
        mcc : float
            Fraction of maternal cells contaminating the sample.
        delta : float
            Sample-level LRR offset from array normalisation.

        Returns
        -------
        gammas : np.ndarray
            Log posterior density of being in each of k hidden states.
        states : list
            Tuple representation of states.
        karyotypes : np.ndarray
            Array of karyotypes in the MetaHMM model.

        """
        alphas, _, states, karyotypes, _ = self.forward_algorithm(
            bafs,
            lrrs,
            sigmas,
            pos,
            haps,
            freqs,
            maternal=maternal,
            pi0=pi0,
            std_dev=std_dev,
            r=r,
            a=a,
            unphased=unphased,
            obs_err=obs_err,
            mcc=mcc,
            delta=delta,
        )
        betas, _, _, _, _ = self.backward_algorithm(
            bafs,
            lrrs,
            sigmas,
            pos,
            haps,
            freqs,
            maternal=maternal,
            pi0=pi0,
            std_dev=std_dev,
            r=r,
            a=a,
            unphased=unphased,
            obs_err=obs_err,
            mcc=mcc,
            delta=delta,
        )
        gammas = (alphas + betas) - logsumexp_sp(alphas + betas, axis=0)
        return gammas, states, karyotypes

    def genome_loglik(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        haps,
        freqs=None,
        maternal=True,
        pi0=0.2,
        std_dev=0.25,
        mcc=0.0,
        delta=0.0,
        **kwargs,
    ):
        """Total log-likelihood of a sample across chromosomes at a shared contamination fraction.

        Each argument from ``bafs`` to ``freqs`` is a list with one entry per
        chromosome, in the form :meth:`forward_algorithm` takes for one
        chromosome. Every chromosome's likelihood is marginalised over all
        karyotypes, so aneuploid chromosomes need no special treatment.

        Parameters
        ----------
        bafs : list of np.ndarray
            Per-chromosome B-allele frequencies.
        lrrs : list of np.ndarray
            Per-chromosome log-R ratios (-9 marks a missing value).
        sigmas : list of np.ndarray
            Per-chromosome per-probe LRR standard deviations.
        pos : list of np.ndarray
            Per-chromosome basepair positions.
        haps : list of np.ndarray
            Per-chromosome 2 x m haplotypes of the observed parent.
        freqs : list of np.ndarray or None
            Per-chromosome allele frequencies for the unobserved parent.
        maternal : bool
            If True, haps are maternal haplotypes; otherwise paternal.
        pi0 : float
            Point-mass weight of the BAF emission, shared across chromosomes.
        std_dev : float
            BAF noise, shared across chromosomes.
        mcc : float
            Fraction of maternal cells contaminating the sample.
        delta : float
            Sample-level LRR offset.
        **kwargs
            Passed to :meth:`forward_algorithm`.

        Returns
        -------
        loglik : float
            Sum of the per-chromosome forward log-likelihoods.

        """
        n = len(bafs)
        assert len(lrrs) == n and len(sigmas) == n
        assert len(pos) == n and len(haps) == n
        if freqs is None:
            freqs = [None] * n
        assert len(freqs) == n
        return sum(
            self.forward_algorithm(
                bafs=bafs[i],
                lrrs=lrrs[i],
                sigmas=sigmas[i],
                pos=pos[i],
                haps=haps[i],
                freqs=freqs[i],
                maternal=maternal,
                pi0=pi0,
                std_dev=std_dev,
                mcc=mcc,
                delta=delta,
                **kwargs,
            )[4]
            for i in range(n)
        )

    def est_delta(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        haps,
        freqs=None,
        maternal=True,
        pi0=0.2,
        std_dev=0.25,
        mcc=0.0,
        delta=0.0,
        max_iter=20,
        tol=1e-4,
        **kwargs,
    ):
        """Estimate the sample-level LRR offset by expectation-maximization at a fixed contamination fraction.

        Given the copy-number posteriors from forward-backward, the LRR term is
        Gaussian in the offset, so the M-step is the precision-weighted mean
        residual between the expected and observed LRR.

        Parameters
        ----------
        bafs, lrrs, sigmas, pos, haps, freqs : list
            Per-chromosome inputs, as in :meth:`genome_loglik`.
        maternal : bool
            If True, haps are maternal haplotypes; otherwise paternal.
        pi0 : float
            Point-mass weight of the BAF emission.
        std_dev : float
            BAF noise.
        mcc : float
            Fraction of maternal cells contaminating the sample.
        delta : float
            Starting value of the offset.
        max_iter : int
            Maximum number of EM iterations.
        tol : float
            Stop once the offset changes by less than this.
        **kwargs
            Passed to :meth:`forward_backward`.

        Returns
        -------
        delta : float
            Estimated LRR offset.
        loglik : float
            Total log-likelihood at the estimated offset.

        """
        assert max_iter > 0
        n = len(bafs)
        if freqs is None:
            freqs = [None] * n
        ks = np.array([sum(s >= 0 for s in state) for state in self.states])
        uniq_ks = np.unique(ks)
        # Expected LRR for each copy number in a contaminated sample, before the offset.
        ratio = ((1.0 - mcc) * uniq_ks + 2.0 * mcc) / 2.0
        mus = np.where(ratio <= 0.125, -3.0, np.log2(np.maximum(ratio, 0.125)))
        for _ in range(max_iter):
            num, den = 0.0, 0.0
            for i in range(n):
                gammas, _, _ = self.forward_backward(
                    bafs=bafs[i],
                    lrrs=lrrs[i],
                    sigmas=sigmas[i],
                    pos=pos[i],
                    haps=haps[i],
                    freqs=freqs[i],
                    maternal=maternal,
                    pi0=pi0,
                    std_dev=std_dev,
                    mcc=mcc,
                    delta=delta,
                    **kwargs,
                )
                post = np.exp(gammas)
                # Posterior mass on each copy number at every site.
                post_k = np.vstack([post[ks == k].sum(axis=0) for k in uniq_ks])
                obs = lrrs[i] != -9
                w = post_k[:, obs] / sigmas[i][obs] ** 2
                num += np.sum(w * (mus[:, None] - lrrs[i][obs]))
                den += np.sum(w)
            if den == 0.0:
                warnings.warn("No observed LRR values; the offset is left unchanged.")
                break
            new_delta = num / den
            converged = abs(new_delta - delta) < tol
            delta = new_delta
            if converged:
                break
        loglik = self.genome_loglik(
            bafs,
            lrrs,
            sigmas,
            pos,
            haps,
            freqs,
            maternal=maternal,
            pi0=pi0,
            std_dev=std_dev,
            mcc=mcc,
            delta=delta,
            **kwargs,
        )
        return delta, loglik

    def est_mcc_genome(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        haps,
        freqs=None,
        maternal=True,
        pi0=None,
        std_dev=None,
        delta=0.0,
        mcc_bounds=(0.0, 0.5),
        n_grid=11,
        alpha=0.05,
        thin=1,
        **kwargs,
    ):
        """Estimate the maternal cell contamination fraction jointly across chromosomes.

        Maximises :meth:`genome_loglik` over a grid of contamination fractions,
        refines the best grid point with a bounded scalar search, and reports
        a likelihood-ratio interval.  BAF noise and the point-mass weight are
        shared across chromosomes; when not given they are estimated from the
        pooled BAF with :meth:`est_sigma_pi0_baf`.

        Parameters
        ----------
        bafs, lrrs, sigmas, pos, haps, freqs : list
            Per-chromosome inputs, as in :meth:`genome_loglik`.
        maternal : bool
            If True, haps are maternal haplotypes; otherwise paternal.
        pi0 : float or None
            Point-mass weight of the BAF emission; estimated when None.
        std_dev : float or None
            BAF noise; estimated when None.
        delta : float or None
            LRR offset.  Fixed at the given value (0 for LRR relative to a
            diploid reference), or estimated at every fraction by
            :meth:`est_delta` when None.
        mcc_bounds : tuple
            Range of contamination fractions searched.
        n_grid : int
            Number of evenly spaced grid points over ``mcc_bounds``.
        alpha : float
            The interval has asymptotic coverage 1 - alpha.
        thin : int
            Fit on every ``thin``-th site of each chromosome (1 uses all
            sites).  Run time falls roughly in proportion.  In simulations,
            ``thin=10`` kept the error in the estimate close to that from
            all sites, but the interval was about three times wider.
        **kwargs
            Passed to :meth:`genome_loglik` (or :meth:`est_delta` when the
            offset is inferred).

        Returns
        -------
        res : dict
            ``mcc`` (estimate), ``mcc_ci`` (interval), ``delta``, ``pi0``,
            ``std_dev``, ``loglik`` (at the estimate, on the sites used),
            ``grid`` / ``grid_loglik`` (the profile over the grid), and ``thin``.

        """
        assert len(mcc_bounds) == 2
        assert 0.0 <= mcc_bounds[0] < mcc_bounds[1] < 1.0
        assert n_grid >= 3
        assert 0.0 < alpha < 1.0
        assert int(thin) == thin and thin >= 1
        if thin > 1:
            # Positions are kept, so transitions span the wider gaps between kept sites.
            bafs, lrrs, sigmas, pos = (
                [x[::thin] for x in v] for v in (bafs, lrrs, sigmas, pos)
            )
            haps = [h[:, ::thin] for h in haps]
            if freqs is not None:
                freqs = [None if f is None else f[::thin] for f in freqs]
        if (pi0 is None) or (std_dev is None):
            pi0_est, sigma_est = self.est_sigma_pi0_baf(np.concatenate(bafs))
            pi0 = pi0_est if pi0 is None else pi0
            std_dev = sigma_est if std_dev is None else std_dev
        infer_delta = delta is None
        shared = dict(maternal=maternal, pi0=pi0, std_dev=std_dev, **kwargs)
        deltas = {}

        def profile(f):
            """Log-likelihood at fraction f, maximised over the offset when it is inferred."""
            if not infer_delta:
                return self.genome_loglik(
                    bafs, lrrs, sigmas, pos, haps, freqs, mcc=f, delta=delta, **shared
                )
            # Warm-start from the offset at the nearest fraction already evaluated.
            start = deltas[min(deltas, key=lambda g: abs(g - f))] if deltas else 0.0
            deltas[f], ll = self.est_delta(
                bafs, lrrs, sigmas, pos, haps, freqs, mcc=f, delta=start, **shared
            )
            return ll

        grid = np.linspace(mcc_bounds[0], mcc_bounds[1], n_grid)
        grid_ll = np.array([profile(f) for f in grid])
        j = int(np.argmax(grid_ll))
        lo, hi = grid[max(j - 1, 0)], grid[min(j + 1, n_grid - 1)]
        opt = minimize_scalar(
            lambda f: -profile(f),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-3},
        )
        mcc_hat, ll_hat = (
            (opt.x, -opt.fun) if -opt.fun > grid_ll[j] else (grid[j], grid_ll[j])
        )
        threshold = ll_hat - chi2.ppf(1.0 - alpha, df=1) / 2.0

        def bound(side):
            """Interval endpoint on one side of the estimate, bracketed by the grid."""
            idx = np.where((grid < mcc_hat) if side < 0 else (grid > mcc_hat))[0]
            below = idx[grid_ll[idx] < threshold]
            if below.size == 0:
                return mcc_bounds[0] if side < 0 else mcc_bounds[1]
            edge = grid[below.max()] if side < 0 else grid[below.min()]
            return brentq(
                lambda f: profile(f) - threshold,
                min(edge, mcc_hat),
                max(edge, mcc_hat),
                xtol=1e-3,
            )

        mcc_ci = (bound(-1), bound(1))
        if infer_delta:
            delta, ll_hat = self.est_delta(
                bafs,
                lrrs,
                sigmas,
                pos,
                haps,
                freqs,
                mcc=mcc_hat,
                delta=deltas[min(deltas, key=lambda g: abs(g - mcc_hat))],
                **shared,
            )
        return {
            "mcc": float(mcc_hat),
            "mcc_ci": (float(mcc_ci[0]), float(mcc_ci[1])),
            "delta": float(delta),
            "pi0": float(pi0),
            "std_dev": float(std_dev),
            "loglik": float(ll_hat),
            "grid": grid,
            "grid_loglik": grid_ll,
            "thin": int(thin),
        }

    def call_genome(
        self,
        bafs,
        lrrs,
        sigmas,
        pos,
        haps,
        freqs=None,
        maternal=True,
        pi0=None,
        std_dev=None,
        delta=0.0,
        thin=1,
        **kwargs,
    ):
        """Estimate contamination across chromosomes, then call each chromosome at that estimate.

        Runs :meth:`est_mcc_genome` and then :meth:`forward_backward` on every
        chromosome at the estimated fraction and offset.  Arguments are as in
        :meth:`est_mcc_genome`; extra keyword arguments go to both.

        Parameters
        ----------
        thin : int
            Estimate the contamination on every ``thin``-th site (see
            :meth:`est_mcc_genome`).  Chromosomes are always called on all
            sites.

        Returns
        -------
        res : dict
            The output of :meth:`est_mcc_genome`, plus ``posteriors`` (one
            :meth:`posterior_karyotypes` dict per chromosome), ``calls`` (the
            maximum-posterior karyotype per chromosome) and ``gammas``.

        """
        res = self.est_mcc_genome(
            bafs,
            lrrs,
            sigmas,
            pos,
            haps,
            freqs,
            maternal=maternal,
            pi0=pi0,
            std_dev=std_dev,
            delta=delta,
            thin=thin,
            **kwargs,
        )
        fb_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ("mcc_bounds", "n_grid", "alpha")
        }
        n = len(bafs)
        if freqs is None:
            freqs = [None] * n
        res["gammas"], res["posteriors"], res["calls"] = [], [], []
        for i in range(n):
            gammas, _, karyotypes = self.forward_backward(
                bafs=bafs[i],
                lrrs=lrrs[i],
                sigmas=sigmas[i],
                pos=pos[i],
                haps=haps[i],
                freqs=freqs[i],
                maternal=maternal,
                pi0=res["pi0"],
                std_dev=res["std_dev"],
                mcc=res["mcc"],
                delta=res["delta"],
                **fb_kwargs,
            )
            post = self.posterior_karyotypes(gammas, karyotypes)
            res["gammas"].append(gammas)
            res["posteriors"].append(post)
            res["calls"].append(str(max(post, key=post.get)))
        return res

    def genotype_parent(
        self,
        bafs,
        lrrs,
        sigmas,
        haps,
        gammas,
        freqs=None,
        maternal=True,
        pi0=0.2,
        std_dev=0.25,
    ):
        """Compute posterior genotype dosages for the unobserved parent.

        For each site integrates the BAF and LRR likelihood over all four possible
        unobserved-parent genotypes (AA, AB, BA, BB), weighted by Hardy-Weinberg
        priors from ``freqs`` and the HMM posterior ``gammas``.  The emissions
        assume an uncontaminated sample with no LRR offset.

        Parameters
        ----------
        bafs : np.ndarray
            m-length array of B-allele frequencies.
        lrrs : np.ndarray
            m-length array of log-R ratios.
        sigmas : np.ndarray
            m-length array of per-site LRR noise estimates.
        haps : np.ndarray
            2 x m array of 0/1 haplotypes for the observed parent.
        gammas : np.ndarray
            k x m log-posterior array from :meth:`forward_backward`.
        freqs : np.ndarray, optional
            m-length array of population allele frequencies; defaults to 0.5 at every site.
        maternal : bool
            If True (default), haps are the maternal haplotypes and the paternal genotype
            is imputed; set False for the reverse.
        pi0 : float
            Sparsity parameter for the BAF emission model.
        std_dev : float
            Noise parameter for the BAF emission model.

        Returns
        -------
        geno_dosage_rev : np.ndarray
            3 x m array of posterior genotype probabilities (log scale) for the unobserved
            parent; rows correspond to homozygous-ref (0), heterozygous (1), and
            homozygous-alt (2).

        """
        assert bafs.ndim == 1
        assert lrrs.ndim == 1
        assert sigmas.ndim == 1
        assert bafs.size == lrrs.size
        assert sigmas.size == lrrs.size
        assert haps.ndim == 2
        assert (pi0 > 0) & (pi0 < 1.0)
        assert std_dev > 0
        assert bafs.size == haps.shape[1]
        assert gammas.ndim == 2
        assert gammas.shape[0] == self.karyotypes.size
        if freqs is not None:
            assert freqs.size == bafs.size
        else:
            freqs = np.repeat(0.5, bafs.size)
        ks = [sum([s >= 0 for s in state]) for state in self.states]
        n = bafs.size
        m = len(self.states)
        geno_dosage = np.zeros(shape=(4, n), dtype=np.float32)
        geno = [[0, 0], [0, 1], [1, 0], [1, 1]]
        for i in range(n):
            f = freqs[i]

            for idx, (x, p) in enumerate(
                zip(geno, ((1 - f) ** 2, f * (1 - f), f * (1 - f), f**2))
            ):
                cur_emissions = np.zeros(m)
                for j in range(m):
                    if maternal:
                        m_ij = mat_dosage(haps[:, i], self.states[j])
                        p_ij = pat_dosage(x, self.states[j])
                    else:
                        m_ij = mat_dosage(x, self.states[j])
                        p_ij = pat_dosage(haps[:, i], self.states[j])
                    # dosage is proportional to likelihood * prior * posterior of specific state
                    cur_emissions[j] = (
                        emission_baf(
                            bafs[i],
                            m_ij,
                            p_ij,
                            pi0=pi0,
                            std_dev=std_dev,
                            k=ks[j],
                        )
                        + emission_lrr(lrrs[i], k=ks[j], std_dev=sigmas[i])
                        + np.log(p)
                        + gammas[j, i]
                    )
                geno_dosage[idx, i] = logsumexp(cur_emissions)
        # Now rescale the dosage estimates here ...
        geno_dosage_rev = np.zeros(shape=(3, n))
        for i in range(n):
            tot = logsumexp_sp(geno_dosage[:, i])
            geno_dosage_rev[0, i] = geno_dosage[0, i] - tot
            geno_dosage_rev[1, i] = (
                logaddexp(geno_dosage[1, i], geno_dosage[2, i]) - tot
            )
            geno_dosage_rev[2, i] = geno_dosage[3, i] - tot
        return geno_dosage_rev

    def infer_missing_af(self, bafs, geno, hap_matrix, pos, eps=0.2):
        """Estimate allele frequencies for the unobserved parent using a haplotype reference panel.

        Identifies sites of opposite homozygosity — where the observed parent is homozygous
        and BAF indicates the unobserved parent carries the opposite allele — then, treating
        adjacent haplotype-row pairs ``(2i, 2i+1)`` in ``hap_matrix`` as the two haplotypes of
        reference individual ``i``, tags every reference *individual* carrying at least one
        copy of that allele. Frequencies at non-anchor sites are estimated via distance-weighted
        interpolation between adjacent anchor sites (Eq. 9 in the PocHMM methods document),
        averaging over **both** haplotypes of each tagged individual — not just the matching one
        — so heterozygous carriers contribute their other, non-matching allele too. Falls back to
        the reference-panel mean at sites with no nearby anchors.

        Parameters
        ----------
        bafs : np.ndarray
            m-length B-allele frequency array.
        geno : np.ndarray
            m-length observed-parent genotype dosage array (0, 1, or 2).
        hap_matrix : np.ndarray
            (2N, m) haplotype reference panel matrix (0/1 encoded). Rows must be paired as
            ``(2i, 2i+1)`` = the two haplotypes of reference individual ``i`` (i.e. an even
            number of rows), since tagging operates at the individual level.
        pos : np.ndarray
            m-length array of genomic positions in basepairs.
        eps : float
            BAF threshold for calling opposite homozygosity (default 0.2).

        Returns
        -------
        freqs : np.ndarray
            m-length array of estimated allele frequencies for the unobserved parent.

        """
        assert bafs.ndim == 1
        assert geno.ndim == 1
        assert hap_matrix.ndim == 2
        assert pos.ndim == 1
        m = bafs.size
        assert geno.size == m
        assert hap_matrix.shape[1] == m
        assert pos.size == m
        assert hap_matrix.shape[0] % 2 == 0
        assert 0.0 < eps < 0.5
        assert np.all(np.isin(geno, [0, 1, 2]))
        assert np.all((bafs >= 0.0) & (bafs <= 1.0))
        assert np.all(pos[1:] > pos[:-1])

        # Fallback: global mean alt allele frequency from the reference panel
        global_freq = hap_matrix.mean(axis=0)

        # Identify anchor sites of opposite homozygosity
        anchor_mask = ((geno == 0) & (bafs >= eps)) | (
            (geno == 2) & (bafs <= 1.0 - eps)
        )
        anchor_indices = np.where(anchor_mask)[0]

        if len(anchor_indices) == 0:
            return global_freq

        # Genotype dosage (0/1/2) per reference individual, pairing rows (2i, 2i+1).
        indiv_geno = hap_matrix[0::2, :] + hap_matrix[1::2, :]

        # Tag reference *individuals* at each anchor by the inferred unobserved-parent allele:
        #   geno==0 (observed hom-ref) → unobserved carries alt → tag individuals with >=1 alt copy
        #   geno==2 (observed hom-alt) → unobserved carries ref → tag individuals with >=1 ref copy
        # Both haplotype rows of each tagged individual are kept (not just the matching one), so
        # heterozygous carriers still contribute their other allele when frequencies are averaged
        # -- otherwise the estimate at the anchor site itself is tautologically 1.0 (or 0.0).
        tagged = {}
        for idx in anchor_indices:
            allele = 1 if geno[idx] == 0 else 0
            carriers = (
                np.where(indiv_geno[:, idx] >= 1)[0]
                if allele == 1
                else np.where(indiv_geno[:, idx] <= 1)[0]
            )
            if len(carriers) > 0:
                tagged[int(idx)] = np.concatenate([carriers * 2, carriers * 2 + 1])

        valid_anchors = sorted(tagged.keys())
        if len(valid_anchors) == 0:
            return global_freq

        va = np.array(valid_anchors)

        def af(anchor_site, query_site):
            """Alt allele frequency at query_site among the haplotypes tagged at anchor_site."""
            return float(hap_matrix[tagged[anchor_site], query_site].mean())

        freqs = np.empty(m)
        for k in range(m):
            ins = int(np.searchsorted(va, k, side="right"))
            left, right = ins - 1, ins

            if left < 0:
                # Before the first anchor: use first anchor's tagged haplotypes
                freqs[k] = af(int(va[0]), k)
            elif right >= len(va):
                # After the last anchor: use last anchor's tagged haplotypes
                freqs[k] = af(int(va[-1]), k)
            elif int(va[left]) == k:
                # At an anchor site: use direct estimate from tagged haplotypes
                freqs[k] = af(k, k)
            else:
                # Between two anchors: linear distance-weighted interpolation
                i, j = int(va[left]), int(va[right])
                d_ij = float(pos[j] - pos[i])
                w_i = 1.0 - float(pos[k] - pos[i]) / d_ij
                w_j = 1.0 - float(pos[j] - pos[k]) / d_ij
                freqs[k] = w_i * af(i, k) + w_j * af(j, k)

        return freqs


class MccEst:
    """Maximum-likelihood estimator of maternal-cell contamination (MCC) from BAF data.

    Maternal-cell contamination occurs when maternal DNA co-purifies with the embryo
    sample, biasing B-allele frequencies towards the maternal genotype.  This class
    provides log-likelihood functions and MLE routines for estimating the contamination
    fraction ``c`` (the proportion of signal attributable to the mother) under two
    data configurations:

    - **Trio** (``_trio`` suffix): both maternal and paternal haplotypes are known, so
      the paternal genotype at each site is observed directly.
    - **POC / duo** (``_poc`` suffix): only maternal haplotypes are available; the
      paternal genotype is marginalised using Hardy–Weinberg allele frequencies.

    Each configuration is further available in an **unphased** form (sites modelled
    independently) and a **phase-aware** form (a 2-state HMM tracks which maternal
    haplotype was transmitted to the embryo).

    Genome-wide inference across multiple chromosomes is supported through
    ``loglik_mcc_genome_*`` and ``est_mcc_genome_*`` methods; per-chromosome
    estimates — useful for identifying aneuploid chromosomes before a genome-wide
    fit — are provided by ``est_mcc_per_chrom_*`` methods.

    All estimation methods jointly optimise the contamination fraction ``c`` and the
    BAF noise standard deviation ``std_dev``.
    """

    def __init__(self):
        """Initialize the MCC estimator (stateless; all state is passed per call)."""
        pass

    def loglik_mcc_poc(self, bafs, mat_haps, freqs, c=0.0, std_dev=0.1):
        """Log-likelihood of MCC for a single chromosome in a mother-child duo (POC).

        At each site the paternal genotype is unknown and is marginalised over
        Hardy-Weinberg frequencies derived from ``freqs``.  The contamination
        fraction ``c`` shifts the expected BAF mean towards the maternal genotype.

        Parameters
        ----------
        bafs : np.ndarray
            m-length array of B-allele frequencies in [0, 1].
        mat_haps : np.ndarray
            2 x m array of 0/1 maternal haplotypes.
        freqs : np.ndarray
            m-length array of population allele frequencies in [0, 1].
        c : float
            Maternal contamination fraction in [0, 0.5].
        std_dev : float
            BAF noise standard deviation (> 0).

        Returns
        -------
        logll : float
            Total log-likelihood summed across all m sites.

        """
        assert bafs.ndim == 1
        assert mat_haps.ndim == 2
        assert bafs.size == mat_haps.shape[1]
        assert freqs.ndim == 1
        assert freqs.size == bafs.size
        assert np.all((bafs >= 0) & (bafs <= 1))
        assert np.all((freqs >= 0) & (freqs <= 1))
        assert (c >= 0) and (c <= 0.5)
        assert std_dev > 0
        mat_geno = np.sum(mat_haps, axis=0).astype(np.int32)
        assert np.all(np.isin(mat_geno, [0, 1, 2]))
        logll = 0.0
        for i in range(bafs.size):
            ll_p0 = 2 * np.log(1.0 - freqs[i]) + loglik_mcc(
                baf=bafs[i], mg=mat_geno[i], pg=0, c=c, std_dev=std_dev
            )
            ll_p1 = np.log(2 * freqs[i] * (1 - freqs[i])) + loglik_mcc(
                baf=bafs[i], mg=mat_geno[i], pg=1, c=c, std_dev=std_dev
            )
            ll_p2 = 2 * np.log(freqs[i]) + loglik_mcc(
                baf=bafs[i], mg=mat_geno[i], pg=2, c=c, std_dev=std_dev
            )
            logll += logsumexp(np.array([ll_p0, ll_p1, ll_p2]))
        return logll

    def loglik_mcc_trio(self, bafs, mat_haps, pat_haps, c=0.0, std_dev=0.1):
        """Log-likelihood of MCC for a single chromosome in a full trio.

        With both parents phased the paternal genotype at every site is known,
        giving a simpler per-site likelihood with no HWE marginalisation.

        Parameters
        ----------
        bafs : np.ndarray
            m-length array of B-allele frequencies in [0, 1].
        mat_haps : np.ndarray
            2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            2 x m array of 0/1 paternal haplotypes.
        c : float
            Maternal contamination fraction in [0, 0.5].
        std_dev : float
            BAF noise standard deviation (> 0).

        Returns
        -------
        logll : float
            Total log-likelihood summed across all m sites.

        """
        assert bafs.ndim == 1
        assert mat_haps.ndim == 2
        assert pat_haps.ndim == 2
        assert bafs.size == mat_haps.shape[1]
        assert bafs.size == pat_haps.shape[1]
        assert np.all((bafs >= 0) & (bafs <= 1))
        assert (c >= 0) and (c <= 0.5)
        assert std_dev > 0
        mat_geno = np.sum(mat_haps, axis=0)
        pat_geno = np.sum(pat_haps, axis=0)
        assert np.all(np.isin(mat_geno, [0, 1, 2]))
        assert np.all(np.isin(pat_geno, [0, 1, 2]))
        logll = 0.0
        for i in range(bafs.size):
            logll += loglik_mcc(
                baf=bafs[i],
                mg=mat_geno[i],
                pg=pat_geno[i],
                c=c,
                std_dev=std_dev,
            )
        return logll

    def est_mcc_poc(self, bafs, mat_haps, freqs, algo="Nelder-Mead", **kwargs):
        """Estimate MCC by MLE for a single chromosome in a mother-child duo.

        Jointly optimises contamination fraction ``c`` and noise ``std_dev`` by
        maximising :meth:`loglik_mcc_poc`.

        Parameters
        ----------
        bafs : np.ndarray
            m-length array of B-allele frequencies.
        mat_haps : np.ndarray
            2 x m array of 0/1 maternal haplotypes.
        freqs : np.ndarray
            m-length array of population allele frequencies.
        algo : str
            Scipy minimisation algorithm; default ``"Nelder-Mead"``.
        **kwargs
            Additional keyword arguments forwarded to ``scipy.optimize.minimize``.

        Returns
        -------
        c_est : float
            MLE of the contamination fraction.
        sigma_est : float
            MLE of the BAF noise standard deviation.

        """
        opt_res = minimize(
            lambda x: (
                -self.loglik_mcc_poc(
                    bafs=bafs,
                    mat_haps=mat_haps,
                    freqs=freqs,
                    c=x[0],
                    std_dev=x[1],
                )
            ),
            x0=[0.05, 0.1],
            method=algo,
            bounds=[(0, 0.5), (1e-3, 0.3)],
            **kwargs,
        )
        c_est = opt_res.x[0]
        sigma_est = opt_res.x[1]
        return c_est, sigma_est

    def est_mcc_trio(self, bafs, mat_haps, pat_haps, algo="Nelder-Mead", **kwargs):
        """Estimate MCC by MLE for a single chromosome in a full trio.

        Jointly optimises contamination fraction ``c`` and noise ``std_dev`` by
        maximising :meth:`loglik_mcc_trio`.

        Parameters
        ----------
        bafs : np.ndarray
            m-length array of B-allele frequencies.
        mat_haps : np.ndarray
            2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            2 x m array of 0/1 paternal haplotypes.
        algo : str
            Scipy minimisation algorithm; default ``"Nelder-Mead"``.
        **kwargs
            Additional keyword arguments forwarded to ``scipy.optimize.minimize``.

        Returns
        -------
        c_est : float
            MLE of the contamination fraction.
        sigma_est : float
            MLE of the BAF noise standard deviation.

        """
        opt_res = minimize(
            lambda x: (
                -self.loglik_mcc_trio(
                    bafs=bafs,
                    mat_haps=mat_haps,
                    pat_haps=pat_haps,
                    c=x[0],
                    std_dev=x[1],
                )
            ),
            x0=[0.05, 0.1],
            method=algo,
            bounds=[(0, 0.5), (1e-3, 0.3)],
            **kwargs,
        )
        c_est = opt_res.x[0]
        sigma_est = opt_res.x[1]
        return c_est, sigma_est

    def mcc_ci_poc(
        self, bafs, mat_haps, freqs, c_hat=0.0, std_dev=0.1, alpha=0.95, df=1
    ):
        """Profile-likelihood confidence interval for MCC in the POC model.

        Uses Wilks' theorem: the CI boundary is the set of ``c`` values where
        twice the log-likelihood drop from the MLE equals the ``alpha`` quantile
        of a chi-squared distribution with ``df`` degrees of freedom.

        Parameters
        ----------
        bafs : np.ndarray
            m-length array of B-allele frequencies.
        mat_haps : np.ndarray
            2 x m array of 0/1 maternal haplotypes.
        freqs : np.ndarray
            m-length array of population allele frequencies.
        c_hat : float
            MLE contamination fraction (the point estimate).
        std_dev : float
            MLE BAF noise standard deviation (held fixed during profile).
        alpha : float
            Confidence level in (0, 1); default 0.95.
        df : int
            Degrees of freedom for chi-squared quantile; default 1.

        Returns
        -------
        lower_CI : float
            Lower confidence bound (falls back to 0 if optimiser fails).
        c_hat : float
            The supplied point estimate.
        upper_CI : float
            Upper confidence bound (falls back to 0.5 if optimiser fails).

        """
        assert (c_hat >= 0) and (c_hat <= 0.5)
        assert std_dev > 0
        assert (alpha > 0) and (alpha < 1)
        wilks = lambda x: (
            2
            * (
                self.loglik_mcc_poc(
                    bafs=bafs, mat_haps=mat_haps, freqs=freqs, c=c_hat, std_dev=std_dev
                )
                - self.loglik_mcc_poc(
                    bafs=bafs, mat_haps=mat_haps, freqs=freqs, c=x, std_dev=std_dev
                )
            )
        )
        qval = chi2.ppf(alpha, df=df)
        try:
            lower_CI = brentq(lambda x: wilks(x) - qval, 1e-4, c_hat)
        except ValueError:
            lower_CI = 0.0
        try:
            upper_CI = brentq(lambda x: wilks(x) - qval, c_hat, 0.5)
        except ValueError:
            upper_CI = 0.5
        return (lower_CI, c_hat, upper_CI)

    def mcc_ci_trio(
        self, bafs, mat_haps, pat_haps, c_hat=0.0, std_dev=0.1, h=1e-5, alpha=0.95, df=1
    ):
        """Profile-likelihood confidence interval for MCC in the trio model.

        Uses Wilks' theorem: the CI boundary is the set of ``c`` values where
        twice the log-likelihood drop from the MLE equals the ``alpha`` quantile
        of a chi-squared distribution with ``df`` degrees of freedom.

        Parameters
        ----------
        bafs : np.ndarray
            m-length array of B-allele frequencies.
        mat_haps : np.ndarray
            2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            2 x m array of 0/1 paternal haplotypes.
        c_hat : float
            MLE contamination fraction (the point estimate).
        std_dev : float
            MLE BAF noise standard deviation (held fixed during profile).
        h : float
            Finite-difference step (unused; retained for API compatibility).
        alpha : float
            Confidence level in (0, 1); default 0.95.
        df : int
            Degrees of freedom for chi-squared quantile; default 1.

        Returns
        -------
        lower_CI : float
            Lower confidence bound (falls back to 0 if optimiser fails).
        c_hat : float
            The supplied point estimate.
        upper_CI : float
            Upper confidence bound (falls back to 0.5 if optimiser fails).

        """
        assert (c_hat >= 0) and (c_hat <= 0.5)
        assert std_dev > 0
        assert (alpha > 0) and (alpha < 1)
        assert h > 0
        wilks = lambda x: (
            2
            * (
                self.loglik_mcc_trio(
                    bafs=bafs,
                    mat_haps=mat_haps,
                    pat_haps=pat_haps,
                    c=c_hat,
                    std_dev=std_dev,
                )
                - self.loglik_mcc_trio(
                    bafs=bafs,
                    mat_haps=mat_haps,
                    pat_haps=pat_haps,
                    c=x,
                    std_dev=std_dev,
                )
            )
        )
        qval = chi2.ppf(alpha, df=df)
        try:
            lower_CI = brentq(lambda x: wilks(x) - qval, 1e-4, c_hat)
        except ValueError:
            lower_CI = 0.0
        try:
            upper_CI = brentq(lambda x: wilks(x) - qval, c_hat, 0.5)
        except ValueError:
            upper_CI = 0.5
        return (lower_CI, c_hat, upper_CI)

    def loglik_mcc_phased_trio(
        self, bafs, mat_haps, pat_haps, pos, c=0.0, std_dev=0.1, r=1e-8
    ):
        """Phase-aware log-likelihood of MCC for a single chromosome in a trio.

        A 2-state HMM tracks which maternal haplotype (0 or 1) was transmitted to
        the embryo.  At each site the emission is a Gaussian centred on the BAF
        expected for the copied maternal allele plus the contamination contribution.
        Transitions follow an exponential recombination kernel:
        ``rho = 1 - exp(-r * d)`` where ``d`` is the inter-site distance in base pairs.

        Parameters
        ----------
        bafs : np.ndarray
            m-length array of B-allele frequencies in [0, 1].
        mat_haps : np.ndarray
            2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            2 x m array of 0/1 paternal haplotypes.
        pos : np.ndarray
            m-length strictly increasing array of base-pair positions.
        c : float
            Maternal contamination fraction in [0, 0.5].
        std_dev : float
            BAF noise standard deviation (> 0).
        r : float
            Per-base-pair recombination rate (> 0); default 1e-8.

        Returns
        -------
        loglik : float
            Total forward-algorithm log-likelihood across m sites.

        """
        assert bafs.ndim == 1
        assert mat_haps.ndim == 2 and mat_haps.shape == (2, bafs.size)
        assert pat_haps.ndim == 2 and pat_haps.shape == (2, bafs.size)
        assert pos.ndim == 1 and pos.size == bafs.size
        assert np.all(pos[1:] > pos[:-1]), "pos must be strictly increasing"
        assert np.all((bafs >= 0) & (bafs <= 1))
        assert (c >= 0) and (c <= 0.5)
        assert std_dev > 0
        assert r > 0
        return forward_mcc_phased_trio(
            bafs.astype(np.float64),
            mat_haps.astype(np.int32),
            pat_haps.astype(np.int32),
            pos.astype(np.float64),
            c=c,
            std_dev=std_dev,
            r=r,
        )

    def loglik_mcc_phased_poc(
        self, bafs, mat_haps, freqs, pos, c=0.0, std_dev=0.1, r=1e-8
    ):
        """Phase-aware log-likelihood of MCC for a single chromosome in a duo (POC).

        Identical to :meth:`loglik_mcc_phased_trio` but marginalises over the
        unobserved paternal genotype at each site using Hardy-Weinberg weights
        derived from ``freqs``.

        Parameters
        ----------
        bafs : np.ndarray
            m-length array of B-allele frequencies in [0, 1].
        mat_haps : np.ndarray
            2 x m array of 0/1 maternal haplotypes.
        freqs : np.ndarray
            m-length array of population allele frequencies in [0, 1].
        pos : np.ndarray
            m-length strictly increasing array of base-pair positions.
        c : float
            Maternal contamination fraction in [0, 0.5].
        std_dev : float
            BAF noise standard deviation (> 0).
        r : float
            Per-base-pair recombination rate (> 0); default 1e-8.

        Returns
        -------
        loglik : float
            Total forward-algorithm log-likelihood across m sites.

        """
        assert bafs.ndim == 1
        assert mat_haps.ndim == 2 and mat_haps.shape == (2, bafs.size)
        assert freqs.ndim == 1 and freqs.size == bafs.size
        assert pos.ndim == 1 and pos.size == bafs.size
        assert np.all(pos[1:] > pos[:-1]), "pos must be strictly increasing"
        assert np.all((bafs >= 0) & (bafs <= 1))
        assert np.all((freqs >= 0) & (freqs <= 1))
        assert (c >= 0) and (c <= 0.5)
        assert std_dev > 0
        assert r > 0
        return forward_mcc_phased_poc(
            bafs.astype(np.float64),
            mat_haps.astype(np.int32),
            freqs.astype(np.float64),
            pos.astype(np.float64),
            c=c,
            std_dev=std_dev,
            r=r,
        )

    def est_mcc_phased_trio(
        self, bafs, mat_haps, pat_haps, pos, r=1e-8, algo="Nelder-Mead", **kwargs
    ):
        """Estimate MCC by MLE using the phase-aware HMM for a single trio chromosome.

        Jointly optimises ``c`` and ``std_dev`` by maximising
        :meth:`loglik_mcc_phased_trio`.

        Parameters
        ----------
        bafs : np.ndarray
            m-length array of B-allele frequencies.
        mat_haps : np.ndarray
            2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            2 x m array of 0/1 paternal haplotypes.
        pos : np.ndarray
            m-length strictly increasing array of base-pair positions.
        r : float
            Per-base-pair recombination rate; default 1e-8.
        algo : str
            Scipy minimisation algorithm; default ``"Nelder-Mead"``.
        **kwargs
            Additional keyword arguments forwarded to ``scipy.optimize.minimize``.

        Returns
        -------
        c_est : float
            MLE of the contamination fraction.
        sigma_est : float
            MLE of the BAF noise standard deviation.

        """
        opt_res = minimize(
            lambda x: (
                -self.loglik_mcc_phased_trio(
                    bafs=bafs,
                    mat_haps=mat_haps,
                    pat_haps=pat_haps,
                    pos=pos,
                    c=x[0],
                    std_dev=x[1],
                    r=r,
                )
            ),
            x0=[0.05, 0.1],
            method=algo,
            bounds=[(0, 0.5), (1e-3, 0.3)],
            **kwargs,
        )
        return opt_res.x[0], opt_res.x[1]

    def est_mcc_phased_poc(
        self, bafs, mat_haps, freqs, pos, r=1e-8, algo="Nelder-Mead", **kwargs
    ):
        """Estimate MCC by MLE using the phase-aware HMM for a single duo (POC) chromosome.

        Jointly optimises ``c`` and ``std_dev`` by maximising
        :meth:`loglik_mcc_phased_poc`.

        Parameters
        ----------
        bafs : np.ndarray
            m-length array of B-allele frequencies.
        mat_haps : np.ndarray
            2 x m array of 0/1 maternal haplotypes.
        freqs : np.ndarray
            m-length array of population allele frequencies.
        pos : np.ndarray
            m-length strictly increasing array of base-pair positions.
        r : float
            Per-base-pair recombination rate; default 1e-8.
        algo : str
            Scipy minimisation algorithm; default ``"Nelder-Mead"``.
        **kwargs
            Additional keyword arguments forwarded to ``scipy.optimize.minimize``.

        Returns
        -------
        c_est : float
            MLE of the contamination fraction.
        sigma_est : float
            MLE of the BAF noise standard deviation.

        """
        opt_res = minimize(
            lambda x: (
                -self.loglik_mcc_phased_poc(
                    bafs=bafs,
                    mat_haps=mat_haps,
                    freqs=freqs,
                    pos=pos,
                    c=x[0],
                    std_dev=x[1],
                    r=r,
                )
            ),
            x0=[0.05, 0.1],
            method=algo,
            bounds=[(0, 0.5), (1e-3, 0.3)],
            **kwargs,
        )
        return opt_res.x[0], opt_res.x[1]

    def mcc_ci_phased_trio(
        self,
        bafs,
        mat_haps,
        pat_haps,
        pos,
        c_hat=0.0,
        std_dev=0.1,
        r=1e-8,
        alpha=0.95,
        df=1,
    ):
        """Profile-likelihood CI for MCC using the phase-aware trio HMM.

        Parameters
        ----------
        bafs : np.ndarray
            m-length array of B-allele frequencies.
        mat_haps : np.ndarray
            2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            2 x m array of 0/1 paternal haplotypes.
        pos : np.ndarray
            m-length strictly increasing array of base-pair positions.
        c_hat : float
            MLE contamination fraction (the point estimate).
        std_dev : float
            MLE BAF noise standard deviation (held fixed during profile).
        r : float
            Per-base-pair recombination rate; default 1e-8.
        alpha : float
            Confidence level in (0, 1); default 0.95.
        df : int
            Degrees of freedom for chi-squared quantile; default 1.

        Returns
        -------
        lower_CI : float
            Lower confidence bound (falls back to 0 if optimiser fails).
        c_hat : float
            The supplied point estimate.
        upper_CI : float
            Upper confidence bound (falls back to 0.5 if optimiser fails).

        """
        assert (c_hat >= 0) and (c_hat <= 0.5)
        assert std_dev > 0
        assert (alpha > 0) and (alpha < 1)
        ll_hat = self.loglik_mcc_phased_trio(
            bafs=bafs,
            mat_haps=mat_haps,
            pat_haps=pat_haps,
            pos=pos,
            c=c_hat,
            std_dev=std_dev,
            r=r,
        )
        wilks = lambda x: (
            2
            * (
                ll_hat
                - self.loglik_mcc_phased_trio(
                    bafs=bafs,
                    mat_haps=mat_haps,
                    pat_haps=pat_haps,
                    pos=pos,
                    c=x,
                    std_dev=std_dev,
                    r=r,
                )
            )
        )
        qval = chi2.ppf(alpha, df=df)
        try:
            lower_CI = brentq(lambda x: wilks(x) - qval, 1e-4, c_hat)
        except ValueError:
            lower_CI = 0.0
        try:
            upper_CI = brentq(lambda x: wilks(x) - qval, c_hat, 0.5)
        except ValueError:
            upper_CI = 0.5
        return (lower_CI, c_hat, upper_CI)

    def mcc_ci_phased_poc(
        self,
        bafs,
        mat_haps,
        freqs,
        pos,
        c_hat=0.0,
        std_dev=0.1,
        r=1e-8,
        alpha=0.95,
        df=1,
    ):
        """Profile-likelihood CI for MCC using the phase-aware POC (duo) HMM.

        Parameters
        ----------
        bafs : np.ndarray
            m-length array of B-allele frequencies.
        mat_haps : np.ndarray
            2 x m array of 0/1 maternal haplotypes.
        freqs : np.ndarray
            m-length array of population allele frequencies.
        pos : np.ndarray
            m-length strictly increasing array of base-pair positions.
        c_hat : float
            MLE contamination fraction (the point estimate).
        std_dev : float
            MLE BAF noise standard deviation (held fixed during profile).
        r : float
            Per-base-pair recombination rate; default 1e-8.
        alpha : float
            Confidence level in (0, 1); default 0.95.
        df : int
            Degrees of freedom for chi-squared quantile; default 1.

        Returns
        -------
        lower_CI : float
            Lower confidence bound (falls back to 0 if optimiser fails).
        c_hat : float
            The supplied point estimate.
        upper_CI : float
            Upper confidence bound (falls back to 0.5 if optimiser fails).

        """
        assert (c_hat >= 0) and (c_hat <= 0.5)
        assert std_dev > 0
        assert (alpha > 0) and (alpha < 1)
        ll_hat = self.loglik_mcc_phased_poc(
            bafs=bafs,
            mat_haps=mat_haps,
            freqs=freqs,
            pos=pos,
            c=c_hat,
            std_dev=std_dev,
            r=r,
        )
        wilks = lambda x: (
            2
            * (
                ll_hat
                - self.loglik_mcc_phased_poc(
                    bafs=bafs,
                    mat_haps=mat_haps,
                    freqs=freqs,
                    pos=pos,
                    c=x,
                    std_dev=std_dev,
                    r=r,
                )
            )
        )
        qval = chi2.ppf(alpha, df=df)
        try:
            lower_CI = brentq(lambda x: wilks(x) - qval, 1e-4, c_hat)
        except ValueError:
            lower_CI = 0.0
        try:
            upper_CI = brentq(lambda x: wilks(x) - qval, c_hat, 0.5)
        except ValueError:
            upper_CI = 0.5
        return (lower_CI, c_hat, upper_CI)

    # ------------------------------------------------------------------
    # Genome-wide log-likelihoods (sum across chromosomes)
    # ------------------------------------------------------------------

    def loglik_mcc_genome_poc(
        self, baf_list, mat_haps_list, freqs_list, c=0.0, std_dev=0.1
    ):
        """Genome-wide log-likelihood of MCC summed across all chromosomes (POC model).

        Chromosomes are conditionally independent given ``c`` and ``std_dev``, so
        the genome-wide log-likelihood is the sum of per-chromosome likelihoods from
        :meth:`loglik_mcc_poc`.  Sex chromosomes should be excluded before calling.

        Implemented by concatenating all chromosome arrays into a single call to
        :meth:`loglik_mcc_poc`, which is equivalent to summing independent
        per-chromosome log-likelihoods but avoids repeated Python function-call
        overhead and assertion checks during optimisation.

        Parameters
        ----------
        baf_list : list of np.ndarray
            Per-chromosome BAF arrays.
        mat_haps_list : list of np.ndarray
            Per-chromosome 2 x m maternal haplotype arrays.
        freqs_list : list of np.ndarray
            Per-chromosome population allele frequency arrays.
        c : float
            Maternal contamination fraction in [0, 0.5].
        std_dev : float
            BAF noise standard deviation (> 0).

        Returns
        -------
        loglik : float
            Total log-likelihood summed across all chromosomes.

        """
        return self.loglik_mcc_poc(
            np.concatenate(baf_list),
            np.concatenate(mat_haps_list, axis=1),
            np.concatenate(freqs_list),
            c=c,
            std_dev=std_dev,
        )

    def loglik_mcc_genome_trio(
        self, baf_list, mat_haps_list, pat_haps_list, c=0.0, std_dev=0.1
    ):
        """Genome-wide log-likelihood of MCC summed across all chromosomes (trio model).

        Implemented by concatenating all chromosome arrays into a single call to
        :meth:`loglik_mcc_trio` (valid because sites are independent across
        chromosomes in the unphased model).

        Parameters
        ----------
        baf_list : list of np.ndarray
            Per-chromosome BAF arrays.
        mat_haps_list : list of np.ndarray
            Per-chromosome 2 x m maternal haplotype arrays.
        pat_haps_list : list of np.ndarray
            Per-chromosome 2 x m paternal haplotype arrays.
        c : float
            Maternal contamination fraction in [0, 0.5].
        std_dev : float
            BAF noise standard deviation (> 0).

        Returns
        -------
        loglik : float
            Total log-likelihood summed across all chromosomes.

        """
        return self.loglik_mcc_trio(
            np.concatenate(baf_list),
            np.concatenate(mat_haps_list, axis=1),
            np.concatenate(pat_haps_list, axis=1),
            c=c,
            std_dev=std_dev,
        )

    def loglik_mcc_genome_phased_poc(
        self, baf_list, mat_haps_list, freqs_list, pos_list, c=0.0, std_dev=0.1, r=1e-8
    ):
        """Genome-wide phase-aware log-likelihood of MCC summed across all chromosomes (POC model).

        Parameters
        ----------
        baf_list : list of np.ndarray
            Per-chromosome BAF arrays.
        mat_haps_list : list of np.ndarray
            Per-chromosome 2 x m maternal haplotype arrays.
        freqs_list : list of np.ndarray
            Per-chromosome population allele frequency arrays.
        pos_list : list of np.ndarray
            Per-chromosome strictly increasing position arrays.
        c : float
            Maternal contamination fraction in [0, 0.5].
        std_dev : float
            BAF noise standard deviation (> 0).
        r : float
            Per-base-pair recombination rate; default 1e-8.

        Returns
        -------
        loglik : float
            Total forward-algorithm log-likelihood summed across all chromosomes.

        """
        return sum(
            self.loglik_mcc_phased_poc(b, m, f, pos, c=c, std_dev=std_dev, r=r)
            for b, m, f, pos in zip(baf_list, mat_haps_list, freqs_list, pos_list)
        )

    def loglik_mcc_genome_phased_trio(
        self,
        baf_list,
        mat_haps_list,
        pat_haps_list,
        pos_list,
        c=0.0,
        std_dev=0.1,
        r=1e-8,
    ):
        """Genome-wide phase-aware log-likelihood of MCC summed across all chromosomes (trio model).

        Parameters
        ----------
        baf_list : list of np.ndarray
            Per-chromosome BAF arrays.
        mat_haps_list : list of np.ndarray
            Per-chromosome 2 x m maternal haplotype arrays.
        pat_haps_list : list of np.ndarray
            Per-chromosome 2 x m paternal haplotype arrays.
        pos_list : list of np.ndarray
            Per-chromosome strictly increasing position arrays.
        c : float
            Maternal contamination fraction in [0, 0.5].
        std_dev : float
            BAF noise standard deviation (> 0).
        r : float
            Per-base-pair recombination rate; default 1e-8.

        Returns
        -------
        loglik : float
            Total forward-algorithm log-likelihood summed across all chromosomes.

        """
        return sum(
            self.loglik_mcc_phased_trio(b, m, p, pos, c=c, std_dev=std_dev, r=r)
            for b, m, p, pos in zip(baf_list, mat_haps_list, pat_haps_list, pos_list)
        )

    # ------------------------------------------------------------------
    # Genome-wide MLE (single c estimated across all chromosomes jointly)
    # ------------------------------------------------------------------

    def est_mcc_genome_poc(
        self, baf_list, mat_haps_list, freqs_list, algo="Nelder-Mead", **kwargs
    ):
        """Estimate MCC by MLE using all chromosomes jointly (POC model).

        Maximises :meth:`loglik_mcc_genome_poc` to obtain a single estimate of
        ``c`` and ``std_dev`` shared across all supplied chromosomes.  Aneuploid
        chromosomes should be excluded from ``baf_list`` before calling; use
        :meth:`est_mcc_per_chrom_poc` to identify outliers first.

        Parameters
        ----------
        baf_list : list of np.ndarray
            Per-chromosome BAF arrays.
        mat_haps_list : list of np.ndarray
            Per-chromosome 2 x m maternal haplotype arrays.
        freqs_list : list of np.ndarray
            Per-chromosome population allele frequency arrays.
        algo : str
            Scipy minimisation algorithm; default ``"Nelder-Mead"``.
        **kwargs
            Additional keyword arguments forwarded to ``scipy.optimize.minimize``.

        Returns
        -------
        c_est : float
            MLE of the contamination fraction.
        sigma_est : float
            MLE of the BAF noise standard deviation.

        """
        opt_res = minimize(
            lambda x: (
                -self.loglik_mcc_genome_poc(
                    baf_list, mat_haps_list, freqs_list, c=x[0], std_dev=x[1]
                )
            ),
            x0=[0.05, 0.1],
            method=algo,
            bounds=[(0, 0.5), (1e-3, 0.3)],
            **kwargs,
        )
        return opt_res.x[0], opt_res.x[1]

    def est_mcc_genome_trio(
        self, baf_list, mat_haps_list, pat_haps_list, algo="Nelder-Mead", **kwargs
    ):
        """Estimate MCC by MLE using all chromosomes jointly (trio model).

        Maximises :meth:`loglik_mcc_genome_trio` to obtain a single estimate of
        ``c`` and ``std_dev`` shared across all supplied chromosomes.

        Parameters
        ----------
        baf_list : list of np.ndarray
            Per-chromosome BAF arrays.
        mat_haps_list : list of np.ndarray
            Per-chromosome 2 x m maternal haplotype arrays.
        pat_haps_list : list of np.ndarray
            Per-chromosome 2 x m paternal haplotype arrays.
        algo : str
            Scipy minimisation algorithm; default ``"Nelder-Mead"``.
        **kwargs
            Additional keyword arguments forwarded to ``scipy.optimize.minimize``.

        Returns
        -------
        c_est : float
            MLE of the contamination fraction.
        sigma_est : float
            MLE of the BAF noise standard deviation.

        """
        opt_res = minimize(
            lambda x: (
                -self.loglik_mcc_genome_trio(
                    baf_list, mat_haps_list, pat_haps_list, c=x[0], std_dev=x[1]
                )
            ),
            x0=[0.05, 0.1],
            method=algo,
            bounds=[(0, 0.5), (1e-3, 0.3)],
            **kwargs,
        )
        return opt_res.x[0], opt_res.x[1]

    def est_mcc_genome_phased_poc(
        self,
        baf_list,
        mat_haps_list,
        freqs_list,
        pos_list,
        r=1e-8,
        algo="Nelder-Mead",
        **kwargs,
    ):
        """Estimate MCC by MLE using all chromosomes jointly (phase-aware POC model).

        Maximises :meth:`loglik_mcc_genome_phased_poc`.

        Parameters
        ----------
        baf_list : list of np.ndarray
            Per-chromosome BAF arrays.
        mat_haps_list : list of np.ndarray
            Per-chromosome 2 x m maternal haplotype arrays.
        freqs_list : list of np.ndarray
            Per-chromosome population allele frequency arrays.
        pos_list : list of np.ndarray
            Per-chromosome strictly increasing position arrays.
        r : float
            Per-base-pair recombination rate; default 1e-8.
        algo : str
            Scipy minimisation algorithm; default ``"Nelder-Mead"``.
        **kwargs
            Additional keyword arguments forwarded to ``scipy.optimize.minimize``.

        Returns
        -------
        c_est : float
            MLE of the contamination fraction.
        sigma_est : float
            MLE of the BAF noise standard deviation.

        """
        opt_res = minimize(
            lambda x: (
                -self.loglik_mcc_genome_phased_poc(
                    baf_list,
                    mat_haps_list,
                    freqs_list,
                    pos_list,
                    c=x[0],
                    std_dev=x[1],
                    r=r,
                )
            ),
            x0=[0.05, 0.1],
            method=algo,
            bounds=[(0, 0.5), (1e-3, 0.3)],
            **kwargs,
        )
        return opt_res.x[0], opt_res.x[1]

    def est_mcc_genome_phased_trio(
        self,
        baf_list,
        mat_haps_list,
        pat_haps_list,
        pos_list,
        r=1e-8,
        algo="Nelder-Mead",
        **kwargs,
    ):
        """Estimate MCC by MLE using all chromosomes jointly (phase-aware trio model).

        Maximises :meth:`loglik_mcc_genome_phased_trio`.

        Parameters
        ----------
        baf_list : list of np.ndarray
            Per-chromosome BAF arrays.
        mat_haps_list : list of np.ndarray
            Per-chromosome 2 x m maternal haplotype arrays.
        pat_haps_list : list of np.ndarray
            Per-chromosome 2 x m paternal haplotype arrays.
        pos_list : list of np.ndarray
            Per-chromosome strictly increasing position arrays.
        r : float
            Per-base-pair recombination rate; default 1e-8.
        algo : str
            Scipy minimisation algorithm; default ``"Nelder-Mead"``.
        **kwargs
            Additional keyword arguments forwarded to ``scipy.optimize.minimize``.

        Returns
        -------
        c_est : float
            MLE of the contamination fraction.
        sigma_est : float
            MLE of the BAF noise standard deviation.

        """
        opt_res = minimize(
            lambda x: (
                -self.loglik_mcc_genome_phased_trio(
                    baf_list,
                    mat_haps_list,
                    pat_haps_list,
                    pos_list,
                    c=x[0],
                    std_dev=x[1],
                    r=r,
                )
            ),
            x0=[0.05, 0.1],
            method=algo,
            bounds=[(0, 0.5), (1e-3, 0.3)],
            **kwargs,
        )
        return opt_res.x[0], opt_res.x[1]

    # ------------------------------------------------------------------
    # Per-chromosome MLE (useful for flagging aneuploid chromosomes)
    # ------------------------------------------------------------------

    def est_mcc_per_chrom_poc(
        self, baf_list, mat_haps_list, freqs_list, algo="Nelder-Mead", **kwargs
    ):
        """Estimate MCC independently per chromosome (POC model).

        Runs :meth:`est_mcc_poc` separately on each chromosome and returns a
        per-chromosome estimate.  Chromosomes whose ``c`` estimate is an outlier
        relative to the genome-wide median are candidates for aneuploidy and should
        be excluded before a joint genome-wide fit with :meth:`est_mcc_genome_poc`.

        Parameters
        ----------
        baf_list : list of np.ndarray
            Per-chromosome BAF arrays.
        mat_haps_list : list of np.ndarray
            Per-chromosome 2 x m maternal haplotype arrays.
        freqs_list : list of np.ndarray
            Per-chromosome population allele frequency arrays.
        algo : str
            Scipy minimisation algorithm; default ``"Nelder-Mead"``.
        **kwargs
            Additional keyword arguments forwarded to ``scipy.optimize.minimize``.

        Returns
        -------
        results : list of tuple
            List of ``(c_hat, std_dev_hat)`` tuples, one per chromosome.

        """
        return [
            self.est_mcc_poc(b, m, f, algo=algo, **kwargs)
            for b, m, f in zip(baf_list, mat_haps_list, freqs_list)
        ]

    def est_mcc_per_chrom_trio(
        self, baf_list, mat_haps_list, pat_haps_list, algo="Nelder-Mead", **kwargs
    ):
        """Estimate MCC independently per chromosome (trio model).

        Runs :meth:`est_mcc_trio` separately on each chromosome.

        Parameters
        ----------
        baf_list : list of np.ndarray
            Per-chromosome BAF arrays.
        mat_haps_list : list of np.ndarray
            Per-chromosome 2 x m maternal haplotype arrays.
        pat_haps_list : list of np.ndarray
            Per-chromosome 2 x m paternal haplotype arrays.
        algo : str
            Scipy minimisation algorithm; default ``"Nelder-Mead"``.
        **kwargs
            Additional keyword arguments forwarded to ``scipy.optimize.minimize``.

        Returns
        -------
        results : list of tuple
            List of ``(c_hat, std_dev_hat)`` tuples, one per chromosome.

        """
        return [
            self.est_mcc_trio(b, m, p, algo=algo, **kwargs)
            for b, m, p in zip(baf_list, mat_haps_list, pat_haps_list)
        ]

    def est_mcc_per_chrom_phased_poc(
        self,
        baf_list,
        mat_haps_list,
        freqs_list,
        pos_list,
        r=1e-8,
        algo="Nelder-Mead",
        **kwargs,
    ):
        """Estimate MCC independently per chromosome (phase-aware POC model).

        Runs :meth:`est_mcc_phased_poc` separately on each chromosome.

        Parameters
        ----------
        baf_list : list of np.ndarray
            Per-chromosome BAF arrays.
        mat_haps_list : list of np.ndarray
            Per-chromosome 2 x m maternal haplotype arrays.
        freqs_list : list of np.ndarray
            Per-chromosome population allele frequency arrays.
        pos_list : list of np.ndarray
            Per-chromosome strictly increasing position arrays.
        r : float
            Per-base-pair recombination rate; default 1e-8.
        algo : str
            Scipy minimisation algorithm; default ``"Nelder-Mead"``.
        **kwargs
            Additional keyword arguments forwarded to ``scipy.optimize.minimize``.

        Returns
        -------
        results : list of tuple
            List of ``(c_hat, std_dev_hat)`` tuples, one per chromosome.

        """
        return [
            self.est_mcc_phased_poc(b, m, f, pos, r=r, algo=algo, **kwargs)
            for b, m, f, pos in zip(baf_list, mat_haps_list, freqs_list, pos_list)
        ]

    def est_mcc_per_chrom_phased_trio(
        self,
        baf_list,
        mat_haps_list,
        pat_haps_list,
        pos_list,
        r=1e-8,
        algo="Nelder-Mead",
        **kwargs,
    ):
        """Estimate MCC independently per chromosome (phase-aware trio model).

        Runs :meth:`est_mcc_phased_trio` separately on each chromosome.

        Parameters
        ----------
        baf_list : list of np.ndarray
            Per-chromosome BAF arrays.
        mat_haps_list : list of np.ndarray
            Per-chromosome 2 x m maternal haplotype arrays.
        pat_haps_list : list of np.ndarray
            Per-chromosome 2 x m paternal haplotype arrays.
        pos_list : list of np.ndarray
            Per-chromosome strictly increasing position arrays.
        r : float
            Per-base-pair recombination rate; default 1e-8.
        algo : str
            Scipy minimisation algorithm; default ``"Nelder-Mead"``.
        **kwargs
            Additional keyword arguments forwarded to ``scipy.optimize.minimize``.

        Returns
        -------
        results : list of tuple
            List of ``(c_hat, std_dev_hat)`` tuples, one per chromosome.

        """
        return [
            self.est_mcc_phased_trio(b, m, p, pos, r=r, algo=algo, **kwargs)
            for b, m, p, pos in zip(baf_list, mat_haps_list, pat_haps_list, pos_list)
        ]


class MosaicEst:
    """Estimator of mosaic copy-number cell fraction using a joint BAF + LRR HMM.

    A 5-state HMM over all m genomic sites estimates the mosaic cell fraction
    ``cf`` and identifies the parental origin of the event.  The five states
    and their emission means are:

    - State 0 (neutral):       LRR = 0,                phased BAF = 0
    - State 1 (maternal gain): LRR = log₂(1 + cf/2),  phased BAF = +cf/6
    - State 2 (paternal gain): LRR = log₂(1 + cf/2),  phased BAF = −cf/6
    - State 3 (maternal loss): LRR = log₂(1 − cf/2),  phased BAF = −cf/2
    - State 4 (paternal loss): LRR = log₂(1 − cf/2),  phased BAF = +cf/2

    The BAF means follow from the discrete mosaic mixture and the phase
    convention (sign = +1 when the maternal haplotype carries the alt allele):

    - Maternal gain: extra maternal copy at a mat=alt site gives BAF 2/3, so
      phased BAF = +cf/6; at mat=ref sites the sign corrects to the same value.
    - Paternal gain: extra paternal copy reverses the BAF shift — phased BAF
      = −cf/6 coherently across all het sites.
    - Maternal loss: losing the maternal copy leaves only paternal alleles,
      giving phased BAF = −cf/2 (3× larger than gain because BAF moves to 0 or 1).
    - Paternal loss: symmetrically, phased BAF = +cf/2.

    LRR is evaluated at **every** site; phased BAF is evaluated only at
    expected-heterozygous sites (one parent hom-ref, the other hom-alt).

    All preprocessing (het-site identification, phase assignment, transition
    matrix construction) happens in ``__init__``, so ``est_mle_cf`` can be
    called immediately after construction.

    Parameters
    ----------
    mat_haps : np.ndarray
        2 × m array of 0/1 maternal haplotypes.
    pat_haps : np.ndarray
        2 × m array of 0/1 paternal haplotypes.
    bafs : np.ndarray
        m-length array of B-allele frequencies.
    pos : np.ndarray
        m-length array of base-pair positions.
    lrrs : np.ndarray, optional
        m-length array of log-R ratio values.  When provided, LRR emission is
        included at all m sites (strongly recommended).
    sigmas : np.ndarray, optional
        m-length array of per-site LRR noise standard deviations.  Required
        when ``lrrs`` is provided.
    switch_err : float
        Within-type origin-switch probability (maternal↔paternal for the same
        gain or loss direction; default 0.01).
    t_rate : float
        Neutral↔aneuploid transition probability (default 1e-4).
    """

    #: Ordered state labels returned by :meth:`infer_origin`.
    STATE_NAMES = (
        "neutral",
        "maternal-gain",
        "paternal-gain",
        "maternal-loss",
        "paternal-loss",
    )

    def __init__(
        self,
        mat_haps,
        pat_haps,
        bafs,
        pos,
        lrrs=None,
        sigmas=None,
        switch_err=0.01,
        t_rate=1e-4,
        **phase_kwargs,
    ):
        """Validate inputs, identify and phase het sites, build transition matrix."""
        assert mat_haps.ndim == 2
        assert pat_haps.ndim == 2
        assert bafs.ndim == 1
        assert bafs.size == mat_haps.shape[1]
        assert bafs.size == pat_haps.shape[1]
        assert pos.size == bafs.size
        if lrrs is not None:
            assert lrrs.ndim == 1 and lrrs.size == bafs.size
        if sigmas is not None:
            assert sigmas.ndim == 1 and sigmas.size == bafs.size
        if lrrs is None:
            warnings.warn(
                "lrrs not provided; LRR emission will be skipped. "
                "Providing per-site LRR substantially improves detection power.",
                UserWarning,
                stacklevel=2,
            )

        self.mat_haps = mat_haps
        self.pat_haps = pat_haps
        self.bafs = bafs
        self.pos = pos
        self.lrrs = lrrs
        self.sigmas = sigmas
        self.mle_cf = None

        self.baf_hets()
        self.phase_hets(**phase_kwargs)
        self.create_transition_matrix(switch_err=switch_err, t_rate=t_rate)

    # ------------------------------------------------------------------
    # Internal preprocessing
    # ------------------------------------------------------------------

    def baf_hets(self):
        """Identify expected-heterozygous sites and cache their BAF values."""
        mat_geno = np.sum(self.mat_haps, axis=0)
        pat_geno = np.sum(self.pat_haps, axis=0)
        exp_het = ((mat_geno == 0) & (pat_geno == 2)) | (
            (mat_geno == 2) & (pat_geno == 0)
        )
        if exp_het.sum() < 10:
            raise ValueError("Fewer than 10 expected heterozygotes found.")
        self.het_idx = np.flatnonzero(exp_het)
        self.het_bafs = self.bafs[exp_het]
        self.n_het = self.het_bafs.size

    def phase_hets(self, **kwargs):
        """Phase BAF at expected-het sites via disomy MetaHMM Viterbi.

        sign = +1 when the maternal haplotype carries the alt allele, −1
        otherwise.  Using sign × (BAF − 0.5) rather than sign × |BAF − 0.5|
        makes gains and losses produce coherent directional shifts across all
        het sites regardless of which parent carries the alt allele.
        """
        meta_hmm = MetaHMM(disomy=True)
        m = self.pos.size
        lrrs_ph = self.lrrs if self.lrrs is not None else np.full(m, -9.0)
        sigmas_ph = self.sigmas if self.sigmas is not None else np.ones(m)
        path, karyo, _, _ = meta_hmm.viterbi_algorithm(
            pos=self.pos,
            mat_haps=self.mat_haps,
            pat_haps=self.pat_haps,
            bafs=self.bafs,
            lrrs=lrrs_ph,
            sigmas=sigmas_ph,
            **kwargs,
        )
        baf_signs = np.empty(self.het_idx.size)
        for i, j in enumerate(self.het_idx):
            baf_signs[i] = 1.0 if self.mat_haps[karyo[path[j]][0], j] == 1 else -1.0
        self.phased_baf = baf_signs * (self.het_bafs - 0.5)

    # ------------------------------------------------------------------
    # Transition matrix (public — allows user tuning after construction)
    # ------------------------------------------------------------------

    def create_transition_matrix(self, switch_err=0.01, t_rate=1e-4):
        """Build the 5-state log-transition matrix.

        States: 0=neutral, 1=mat-gain, 2=pat-gain, 3=mat-loss, 4=pat-loss.

        Transitions allowed:
        - neutral → each aneuploid state with probability ``t_rate / 4``
        - any aneuploid → neutral with probability ``t_rate``
        - within-gain origin switch (1↔2) with probability ``switch_err``
        - within-loss origin switch (3↔4) with probability ``switch_err``
        - cross-type transitions (gain↔loss) are not modelled (log-prob = −∞)

        Parameters
        ----------
        switch_err : float
            Maternal-paternal origin-switch probability within the same gain or loss direction.
        t_rate : float
            Neutral-aneuploid entry/exit probability.
        """
        assert 0 < switch_err <= 0.05
        assert 0 < t_rate < 0.5
        n = 5
        A = np.zeros((n, n))
        # Neutral → each aneuploid equally
        A[0, 1:] = t_rate / 4.0
        # Each aneuploid → neutral
        A[1:, 0] = t_rate
        # Within-gain origin switch
        A[1, 2] = switch_err
        A[2, 1] = switch_err
        # Within-loss origin switch
        A[3, 4] = switch_err
        A[4, 3] = switch_err
        # Diagonal
        for i in range(n):
            A[i, i] = 1.0 - A[i, :].sum()
        # Cross-type transitions (gain↔loss) are 0; log(0) = -inf is intentional
        with np.errstate(divide="ignore"):
            self.A = np.log(A)

    # ------------------------------------------------------------------
    # Likelihood and inference
    # ------------------------------------------------------------------

    def forward_algo_full(self, cf=0.0, std_dev_baf=0.1):
        """Forward algorithm over all m sites with joint BAF + LRR emission.

        When ``lrrs`` was provided at construction, LRR emission is added at
        every site.  Phased BAF emission is always added at expected-het sites.

        State-dependent emission means (see class docstring for derivation):

        - State 0 (neutral):       LRR = 0,                phased BAF = 0
        - State 1 (maternal gain): LRR = log₂(1 + cf/2),  phased BAF = +cf/6
        - State 2 (paternal gain): LRR = log₂(1 + cf/2),  phased BAF = −cf/6
        - State 3 (maternal loss): LRR = log₂(1 − cf/2),  phased BAF = −cf/2
        - State 4 (paternal loss): LRR = log₂(1 − cf/2),  phased BAF = +cf/2

        Parameters
        ----------
        cf : float
            Mosaic cell fraction in [0, 1).
        std_dev_baf : float
            BAF noise standard deviation at het sites.

        Returns
        -------
        alphas : np.ndarray
            5 x m log forward variable.
        scaler : np.ndarray
            m-length per-site log normalisation constants.
        loglik : float
            Total log-likelihood.
        """
        from scipy.stats import norm as _norm

        assert 0.0 <= cf < 1.0
        assert std_dev_baf > 0.0

        m = self.pos.size
        use_lrr = self.lrrs is not None and self.sigmas is not None
        n_states = 5

        lrr_gain = np.log2(1.0 + cf / 2.0)
        lrr_loss = np.log2(max(1.0 - cf / 2.0, 1e-9))
        # Per-state LRR means: [neutral, mat-gain, pat-gain, mat-loss, pat-loss]
        lrr_means = np.array([0.0, lrr_gain, lrr_gain, lrr_loss, lrr_loss])
        # Per-state phased-BAF means
        baf_means = np.array([0.0, cf / 6.0, -cf / 6.0, -cf / 2.0, cf / 2.0])

        if use_lrr:
            lrr_logp = np.stack(
                [
                    _norm.logpdf(self.lrrs, lrr_means[k], self.sigmas)
                    for k in range(n_states)
                ]
            )
        baf_logp = np.stack(
            [
                _norm.logpdf(self.phased_baf, baf_means[k], std_dev_baf)
                for k in range(n_states)
            ]
        )

        is_het = np.zeros(m, dtype=bool)
        is_het[self.het_idx] = True
        het_to_pbaf = np.full(m, -1, dtype=np.intp)
        het_to_pbaf[self.het_idx] = np.arange(self.het_idx.size)

        alphas = np.zeros((n_states, m))
        alphas[:, 0] = np.log(1.0 / n_states)
        if use_lrr:
            alphas[:, 0] += lrr_logp[:, 0]
        if is_het[0]:
            alphas[:, 0] += baf_logp[:, het_to_pbaf[0]]

        scaler = np.zeros(m)
        scaler[0] = logsumexp(alphas[:, 0])
        alphas[:, 0] -= scaler[0]

        for i in range(1, m):
            for k in range(n_states):
                alphas[k, i] = logsumexp(self.A[:, k] + alphas[:, i - 1])
            if use_lrr:
                alphas[:, i] += lrr_logp[:, i]
            if is_het[i]:
                alphas[:, i] += baf_logp[:, het_to_pbaf[i]]
            scaler[i] = logsumexp(alphas[:, i])
            alphas[:, i] -= scaler[i]

        return alphas, scaler, float(np.sum(scaler))

    def est_mle_cf(self, std_dev_baf=0.1):
        """Estimate the MLE mosaic cell fraction.

        Maximises :meth:`forward_algo_full` log-likelihood over cf in [0, 1)
        using Powell's method.  Stores the result in ``self.mle_cf``
        (``nan`` on optimisation failure).

        Parameters
        ----------
        std_dev_baf : float
            BAF noise standard deviation.
        """
        try:
            f = lambda x: (
                -self.forward_algo_full(cf=float(x[0]), std_dev_baf=std_dev_baf)[2]
            )
            opt_res = minimize(
                f, x0=[0.1], method="Powell", tol=1e-5, bounds=[(0.0, 0.999)]
            )
            self.mle_cf = float(opt_res.x[0])
        except (ValueError, AssertionError):
            self.mle_cf = np.nan

    def ci_mle_cf(self, std_dev_baf=0.1, h=1e-4):
        """95% confidence interval for ``mle_cf`` via observed Fisher information.

        Requires :meth:`est_mle_cf` to have been called first.

        Parameters
        ----------
        std_dev_baf : float
            BAF noise standard deviation.
        h : float
            Finite-difference step size.

        Returns
        -------
        ci : list
            ``[lower_95, mle_cf, upper_95]``, clamped to [0, 1].
        """
        assert self.mle_cf is not None and not np.isnan(self.mle_cf)
        ci = [np.nan, np.nan, np.nan]
        try:
            f = lambda x: self.forward_algo_full(cf=x, std_dev_baf=std_dev_baf)[2]
            cf = self.mle_cf
            if cf < h:
                logI = (f(cf + h) - 2.0 * f(cf) + f(cf)) / h**2
            elif cf > (0.999 - h):
                logI = (f(cf) - 2.0 * f(cf) + f(cf - h)) / h**2
            else:
                logI = (f(cf + h) - 2.0 * f(cf) + f(cf - h)) / h**2
            se = np.sqrt(1.0 / (-logI))
            ci[0] = max(0.0, cf - 1.96 * se)
            ci[1] = cf
            ci[2] = min(1.0, cf + 1.96 * se)
        except (ValueError, ZeroDivisionError):
            pass
        return ci

    def lrt_cf(self, std_dev_baf=0.1):
        """Likelihood-ratio test statistic for cf > 0 vs cf = 0.

        Computes -2 x (l(cf=0) - l(cf_MLE)).  Under the null hypothesis of
        no mosaicism this is approximately chi-squared with 1 degree of
        freedom.  Calls :meth:`est_mle_cf` if it has not been run yet.

        Parameters
        ----------
        std_dev_baf : float
            BAF noise standard deviation.

        Returns
        -------
        lrt : float
            LRT statistic, or ``nan`` on failure.
        """
        ll_h0 = self.forward_algo_full(cf=0.0, std_dev_baf=std_dev_baf)[2]
        if self.mle_cf is None:
            self.est_mle_cf(std_dev_baf=std_dev_baf)
        if not np.isnan(self.mle_cf):
            ll_h1 = self.forward_algo_full(cf=self.mle_cf, std_dev_baf=std_dev_baf)[2]
            return -2.0 * (ll_h0 - ll_h1)
        return np.nan

    def infer_origin(self, std_dev_baf=0.1):
        """Identify the most likely parental origin of the mosaic event.

        Runs the forward algorithm at ``mle_cf`` and computes the
        chromosome-wide log-evidence for each of the four aneuploid states
        (marginalised over sites via logsumexp of the forward variable).  The
        aneuploid state with the highest evidence is returned as the inferred
        origin.

        Requires :meth:`est_mle_cf` to have been called first.  Returns
        ``'neutral'`` when ``mle_cf`` is below 0.01.

        Parameters
        ----------
        std_dev_baf : float
            BAF noise standard deviation.

        Returns
        -------
        origin : str
            One of ``MosaicEst.STATE_NAMES``.
        """
        assert self.mle_cf is not None and not np.isnan(self.mle_cf)
        if self.mle_cf < 0.01:
            return "neutral"
        alphas, _, _ = self.forward_algo_full(cf=self.mle_cf, std_dev_baf=std_dev_baf)
        # Chromosome-wide log-evidence per state via logsumexp across sites
        log_evidence = np.array([logsumexp(alphas[k, :]) for k in range(5)])
        # Select the aneuploid state (1-4) with highest evidence
        best = int(np.argmax(log_evidence[1:])) + 1
        return self.STATE_NAMES[best]


class PhaseCorrect:
    """Module for implementing Mendelian phase correction using BAF data.

    Implements the key methods for phasing parental haplotypes using Mendelian phasing via a LOD-score

    """

    def __init__(self, mat_haps, pat_haps, pos):
        """Initialize the class for phase correction.

        Parameters
        ----------
        mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.
        pos : np.ndarray
            An m-length array of base-pair positions.

        """
        assert mat_haps.shape[0] == pat_haps.shape[0]
        assert mat_haps.shape[1] == pat_haps.shape[1]
        assert np.all(np.isin(mat_haps, [0, 1]))
        assert np.all(np.isin(mat_haps, [0, 1]))
        assert pos.ndim == 1
        assert pos.size == mat_haps.shape[1]
        self.mat_haps = mat_haps
        self.pat_haps = pat_haps
        self.pos = pos
        self.mat_haps_true = None
        self.pat_haps_true = None
        self.mat_haps_fixed = None
        self.pat_haps_fixed = None
        self.embryo_bafs = None
        self.embryo_pi0s = None
        self.embryo_sigmas = None

    def add_true_haps(self, true_mat_haps, true_pat_haps):
        """Add in true haplotypes if available from a simulation.

        Parameters
        ----------
        true_mat_haps : np.ndarray
            A 2 x m array of 0/1 maternal haplotypes.
        true_pat_haps : np.ndarray
            A 2 x m array of 0/1 paternal haplotypes.

        """
        assert true_mat_haps.ndim == 2
        assert true_pat_haps.ndim == 2
        assert true_mat_haps.shape[0] == self.mat_haps.shape[0]
        assert true_mat_haps.shape[1] == self.mat_haps.shape[1]
        assert true_pat_haps.shape[0] == self.pat_haps.shape[0]
        assert true_pat_haps.shape[1] == self.pat_haps.shape[1]
        self.mat_haps_true = true_mat_haps
        self.pat_haps_true = true_pat_haps

    def add_baf(self, embryo_bafs=[]):
        """Add in BAF estimates for each embryo.

        Parameters
        ----------
        embryo_bafs : list
            A list of per-embryo BAF arrays from the same parental individuals.

        """
        assert len(embryo_bafs) > 0
        self.embryo_bafs = []
        for baf in embryo_bafs:
            assert baf.size == self.mat_haps.shape[1]
            assert baf.size == self.pat_haps.shape[1]
            self.embryo_bafs.append(baf)

    def est_sigma_pi0s(self, **kwargs):
        """Estimate per-embryo BAF noise parameters under a disomy assumption.

        Runs :meth:`MetaHMM.est_sigma_pi0` (disomy model) on each embryo BAF
        array and stores the results in ``self.embryo_pi0s`` and
        ``self.embryo_sigmas``.  Must be called before :meth:`viterbi_phase_correct`
        or :meth:`flag_parental_genotype_errors`.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`MetaHMM.est_sigma_pi0`
            (e.g. ``pi0_bounds``, ``sigma_bounds``, ``algo``).
        """
        assert self.embryo_bafs is not None
        hmm = MetaHMM(disomy=True)
        m = self.pos.size
        lrrs_missing = np.full(m, -9.0)
        sigmas_lrr = np.ones(m)
        pi0_est_acc = np.zeros(len(self.embryo_bafs))
        sigma_est_acc = np.zeros(len(self.embryo_bafs))
        for i, baf in enumerate(self.embryo_bafs):
            pi0_est, sigma_est = hmm.est_sigma_pi0(
                baf,
                self.pos,
                self.mat_haps,
                self.pat_haps,
                lrrs=lrrs_missing,
                sigmas=sigmas_lrr,
                **kwargs,
            )
            pi0_est_acc[i] = pi0_est
            sigma_est_acc[i] = sigma_est
        self.embryo_pi0s = pi0_est_acc
        self.embryo_sigmas = sigma_est_acc

    def correct_haps_viterbi(self, haps, paths):
        """Correct haplotype phase using majority-rule consensus across copying paths.

        At each inter-site transition the number of siblings showing a haplotype
        switch is counted.  If a majority of siblings switch, the haplotype
        orientations prior to that site are flipped for both strands.

        Parameters
        ----------
        haps : np.ndarray
            2 x m array of 0/1 haplotypes to be corrected.
        paths : np.ndarray
            n_sib x m boolean array where entry ``[j, i]`` is 1 if sibling j
            copies haplotype 0 at site i (output of Viterbi decoding).

        Returns
        -------
        n_mis : np.ndarray
            (m-1)-length array of per-interval mismatch counts.
        fixed_haps : np.ndarray
            2 x m phase-corrected haplotype array.

        """
        assert paths.shape[1] == haps.shape[1]
        assert paths.shape[0] > 1
        n_sib = paths.shape[0]
        m = haps.shape[1]
        n_mis = np.zeros(m - 1)
        for i, j in zip(np.arange(m - 1), np.arange(1, m)):
            n_mis[i] = np.sum((paths[:, i] != paths[:, j]))
        hap_idx1 = np.zeros(haps.shape[1], dtype=np.uint16)
        hap_idx2 = np.ones(haps.shape[1], dtype=np.uint16)
        for i in np.arange(m - 1):
            # If majority have the same switch - we swap haplotypes ...
            # n_mis[i] covers the interval (i, i+1), so the switch sits between
            # them and site i belongs on the near side: the slice must be i + 1.
            if n_mis[i] > (n_sib / 2):
                hap_idx1[: i + 1] = 1 - hap_idx1[: i + 1]
                hap_idx2[: i + 1] = 1 - hap_idx2[: i + 1]
        fixed_hap1 = [haps[x, i] for i, x in enumerate(hap_idx1)]
        fixed_hap2 = [haps[x, i] for i, x in enumerate(hap_idx2)]
        fixed_haps = np.vstack([fixed_hap1, fixed_hap2])
        assert fixed_haps.shape == haps.shape
        return n_mis, fixed_haps

    def viterbi_phase_correct(self, niter=5, r=1e-8):
        """Use the Viterbi decoding for phase correction in multiple iterations.

        Parameters
        ----------
        niter : int
            Number of iterations for Viterbi-based phase correction.
        r : float
            Recombination rate per base pair.

        Returns
        -------
        mat_haps : np.ndarray
            Phase-corrected 2 x m maternal haplotype array.
        pat_haps : np.ndarray
            Phase-corrected 2 x m paternal haplotype array.
        n_mis_mat_tot : np.ndarray
            niter-length array of total maternal mismatch counts per iteration.
        n_mis_pat_tot : np.ndarray
            niter-length array of total paternal mismatch counts per iteration.

        """
        assert niter > 0
        hmm = MetaHMM(disomy=True)
        mat_haps = self.mat_haps
        pat_haps = self.pat_haps
        n_sibs = len(self.embryo_bafs)
        m = self.pos.size
        lrrs_missing = np.full(m, -9.0)
        sigmas_lrr = np.ones(m)
        n_mis_mat_tot = np.zeros(niter)
        n_mis_pat_tot = np.zeros(niter)
        for i in range(niter):
            mat_paths = np.zeros(shape=(n_sibs, m))
            pat_paths = np.zeros(shape=(n_sibs, m))
            for j, baf in enumerate(self.embryo_bafs):
                path, _, _, _ = hmm.viterbi_algorithm(
                    baf,
                    lrrs_missing,
                    sigmas_lrr,
                    self.pos,
                    mat_haps,
                    pat_haps,
                    r=r,
                    pi0=self.embryo_pi0s[j],
                    std_dev=self.embryo_sigmas[j],
                )
                # Collect indicators of both copying the same haplotype
                mat_paths[j, :] = np.isin(path, [0, 1])
                pat_paths[j, :] = np.isin(path, [0, 2])
            # now we apply the minimum recombination path-switching setting ...
            n_mis_mat, fixed_mat_haps = self.correct_haps_viterbi(mat_haps, mat_paths)
            n_mis_pat, fixed_pat_haps = self.correct_haps_viterbi(pat_haps, pat_paths)
            n_mis_mat_tot[i] = np.sum(n_mis_mat)
            n_mis_pat_tot[i] = np.sum(n_mis_pat)
            # NOTE: print out that the mismatches have been minimized on both paternal & maternal chromosomes this way?
            mat_haps = fixed_mat_haps
            pat_haps = fixed_pat_haps
        # Set the corrected haplotypes here ...
        self.mat_haps_fixed = mat_haps
        self.pat_haps_fixed = pat_haps
        return mat_haps, pat_haps, n_mis_mat_tot, n_mis_pat_tot

    def flag_parental_genotype_errors(self, use_fixed=False, r=1e-8):
        """Flag potential parental genotype errors using multiple euploid siblings.

        Runs forward-backward under disomy for each sibling embryo and accumulates
        the posterior-weighted log Bayes factor error scores across all siblings.
        Genuine genotype errors are consistently flagged across siblings (same
        parental genotype), while per-embryo noise is uncorrelated and averages out.

        Parameters
        ----------
        use_fixed : bool
            Use phase-corrected haplotypes if available (default: False).
        r : float
            Recombination rate per base pair.

        Returns
        -------
        mat_err_scores : np.ndarray
            m-length array of summed maternal error scores.
        pat_err_scores : np.ndarray
            m-length array of summed paternal error scores.

        """
        assert self.embryo_bafs is not None
        assert self.embryo_pi0s is not None, "Call est_sigma_pi0s first"
        if use_fixed:
            assert self.mat_haps_fixed is not None, "Call viterbi_phase_correct first"
            mat_haps = self.mat_haps_fixed
            pat_haps = self.pat_haps_fixed
        else:
            mat_haps = self.mat_haps
            pat_haps = self.pat_haps
        m = self.pos.size
        lrrs_missing = np.full(m, -9.0)
        sigmas_lrr = np.ones(m)
        hmm = MetaHMM(disomy=True)
        mat_err_scores = np.zeros(m)
        pat_err_scores = np.zeros(m)
        for j, baf in enumerate(self.embryo_bafs):
            gammas, states, _ = hmm.forward_backward(
                baf,
                lrrs_missing,
                sigmas_lrr,
                self.pos,
                mat_haps,
                pat_haps,
                pi0=self.embryo_pi0s[j],
                std_dev=self.embryo_sigmas[j],
                r=r,
            )
            mat_err_j, pat_err_j = hmm.flag_parental_genotype_errors(
                gammas,
                states,
                baf,
                mat_haps,
                pat_haps,
                pi0=self.embryo_pi0s[j],
                std_dev=self.embryo_sigmas[j],
            )
            mat_err_scores += mat_err_j
            pat_err_scores += pat_err_j
        return mat_err_scores, pat_err_scores

    def estimate_switch_err_true(self, maternal=True, fixed=False):
        """Estimate the switch error from true and inferred haplotypes.

        The switch error is defined as consecutive heterozygotes that are
        in the incorrect orientation.

        Parameters
        ----------
        maternal : bool
            Apply the function to the maternal haplotypes (default: True).
        fixed : bool
            Apply the function to the fixed haplotypes (default: False).

        Returns
        -------
        n_switches : int
            Number of switches between consecutive heterozygotes.
        n_consecutive_hets : int
            Number of consecutive heterozygotes.
        switch_err_rate : float
            Number of switches per consecutive heterozygote.
        switch_idx : list
            SNP pairs where the variant is out of phase with its predecessor.
        het_idx : np.ndarray
            Locations/indices of the heterozygotes.
        lods : None
            LOD-scores for the estimated switches (currently always None).

        """
        assert self.mat_haps_true is not None
        assert self.pat_haps_true is not None
        if maternal:
            true_haps = self.mat_haps_true
            if fixed:
                assert self.mat_haps_fixed is not None
                inf_haps = self.mat_haps_fixed
            else:
                inf_haps = self.mat_haps
        else:
            true_haps = self.pat_haps_true
            if fixed:
                assert self.pat_haps_fixed is not None
                inf_haps = self.pat_haps_fixed
            else:
                inf_haps = self.pat_haps
        # NOTE: this is just between all consecutive hets ...
        geno = true_haps.sum(axis=0)
        het_idxs = np.where(geno == 1)[0]
        n_switches = 0
        n_consecutive_hets = 0
        switch_idxs = []
        for i, j in zip(het_idxs[:-1], het_idxs[1:]):
            assert inf_haps[:, i].sum() == 1
            assert inf_haps[:, j].sum() == 1
            n_consecutive_hets += 1
            true_hap = true_haps[:, [i, j]]
            inf_hap = inf_haps[:, [i, j]]
            # Check if the heterozygotes are oriented appropriately
            if ~(
                np.all(true_hap[0, :] == inf_hap[0, :])
                or np.all(true_hap[0, :] == inf_hap[1, :])
            ):
                n_switches += 1
                switch_idxs.append((i, j))
        return (
            n_switches,
            n_consecutive_hets,
            n_switches / n_consecutive_hets,
            switch_idxs,
            het_idxs,
            None,
        )


class RecombEst(PhaseCorrect):
    """Crossover detection via allele-transmission comparison across sibling embryos.

    Implements the simplified crossover-detection approach of Coop et al. (2007),
    extended with :class:`QuadHMM`-based interval refinement.  For each pair of
    (template, non-template) sibling embryos, the log-likelihood ratio of copying
    the *same* vs *different* parental allele is computed at informative markers
    (one parent heterozygous, the other homozygous).  Crossover locations are
    identified as positions where the sign of this ratio changes consistently
    across sibling pairs.

    Inherits the embryo storage, noise-parameter estimation, and phase-correction
    infrastructure from :class:`PhaseCorrect`.
    """

    def __init__(self, **kwargs):
        """Initialize RecombEst inheriting from PhaseCorrect.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`PhaseCorrect.__init__`
            (``mat_haps``, ``pat_haps``, ``pos``).
        """
        super(RecombEst, self).__init__(**kwargs)

    def informative_markers(self, maternal=True):
        """Return a boolean mask of allele-transmission-informative sites.

        A site is informative for the maternal haplotype when the mother is
        heterozygous (genotype 1) and the father is homozygous (genotype 0 or 2),
        so the transmitted maternal allele is unambiguously determined by the
        embryo BAF.  The paternal case is the mirror image.

        Parameters
        ----------
        maternal : bool
            If True (default) return sites informative for maternal crossovers;
            if False return sites informative for paternal crossovers.

        Returns
        -------
        info_idx : np.ndarray
            m-length boolean mask; True at informative sites.

        """
        mat_geno = np.sum(self.mat_haps, axis=0)
        pat_geno = np.sum(self.pat_haps, axis=0)
        if maternal:
            info_idx = ((mat_geno == 1) & (pat_geno == 0)) | (
                (mat_geno == 1) & (pat_geno == 2)
            )
        else:
            info_idx = ((pat_geno == 1) & (mat_geno == 0)) | (
                (pat_geno == 1) & (mat_geno == 2)
            )
        return info_idx

    def refine_recomb_events(self, potential_switches, npad=5):
        """Refine recombination estimation using the switch-clusters approach of Coop et al 2007.

        Parameters
        ----------
        potential_switches : np.ndarray
            Array of potential switch indices at informative markers.
        npad : int
            Half-width (in informative SNPs) of the switch cluster window.

        Returns
        -------
        subset_co : list
            Isolated crossover indices surviving the switch-cluster filter.

        """
        assert npad > 1
        if potential_switches.size > 0:
            subset_co = []
            for i in potential_switches:
                # Count the number of switches within 5 informative SNPs
                n = np.sum(
                    (potential_switches < i + npad) & (potential_switches > i - npad)
                )
                # Even number of crossovers suggest it is likely not a well-supported CO
                if n % 2 == 1:
                    subset_co.append(i)
            return subset_co
        else:
            return []

    def isolate_recomb_events(
        self, template_embryo=0, maternal=True, ll_thresh=0, npad=5
    ):
        """Isolate specific recombination events.

        Parameters
        ----------
        template_embryo : int
            Index of the template embryo.
        maternal : bool
            If True, estimate maternal crossovers; if False, estimate paternal crossovers.
        ll_thresh : float
            Log-likelihood ratio threshold for transmission assignment.
        npad : int
            Half-width (in informative SNPs) of the switch cluster window.

        Returns
        -------
        Z : np.ndarray
            n_sib x n_info array of switch indicators for allele copying in non-template embryos.
        llr_z : np.ndarray
            n_sib x n_info raw LLR values for allele-copying in non-template embryos.
        potential_switches_filt : np.ndarray
            Majority-vote filtered switch indices at informative markers.
        switch_cnts_filt : np.ndarray
            Support counts for each filtered switch.

        """
        assert self.embryo_bafs is not None
        assert self.embryo_pi0s is not None
        assert self.embryo_sigmas is not None
        m = len(self.embryo_bafs)
        assert m > 1
        assert template_embryo < m
        assert ll_thresh >= 0
        non_template_ids = [i for i in range(m) if i != template_embryo]
        # Get the informative snps for the specific parent
        info_snps = self.informative_markers(maternal=maternal)
        llr_z = np.zeros(shape=(len(non_template_ids), np.sum(info_snps)))
        for j, k in enumerate(non_template_ids):
            z1s = np.zeros(np.sum(info_snps))
            z2s = np.zeros(np.sum(info_snps))
            # NOTE: we use a signed-version of the log-likelihood ratio ...
            for p, i in enumerate(np.where(info_snps)[0]):
                z_same = np.zeros(2)
                z_diff = np.zeros(2)
                if maternal:
                    # Same allele is transmitted
                    z_same[0] = emission_baf(
                        self.embryo_bafs[template_embryo][i],
                        m=0,
                        p=self.pat_haps[0, i],
                        pi0=self.embryo_pi0s[template_embryo],
                        std_dev=self.embryo_sigmas[template_embryo],
                    ) + emission_baf(
                        self.embryo_bafs[k][i],
                        m=0,
                        p=self.pat_haps[0, i],
                        pi0=self.embryo_pi0s[k],
                        std_dev=self.embryo_sigmas[k],
                    )
                    z_same[1] = emission_baf(
                        self.embryo_bafs[template_embryo][i],
                        m=1,
                        p=self.pat_haps[0, i],
                        pi0=self.embryo_pi0s[template_embryo],
                        std_dev=self.embryo_sigmas[template_embryo],
                    ) + emission_baf(
                        self.embryo_bafs[k][i],
                        m=1,
                        p=self.pat_haps[0, i],
                        pi0=self.embryo_pi0s[k],
                        std_dev=self.embryo_sigmas[k],
                    )
                    # Same allele is not transmitted ...
                    z_diff[0] = emission_baf(
                        self.embryo_bafs[template_embryo][i],
                        m=0,
                        p=self.pat_haps[0, i],
                        pi0=self.embryo_pi0s[template_embryo],
                        std_dev=self.embryo_sigmas[template_embryo],
                    ) + emission_baf(
                        self.embryo_bafs[k][i],
                        m=1,
                        p=self.pat_haps[0, i],
                        pi0=self.embryo_pi0s[k],
                        std_dev=self.embryo_sigmas[k],
                    )
                    z_diff[1] = emission_baf(
                        self.embryo_bafs[template_embryo][i],
                        m=1,
                        p=self.pat_haps[0, i],
                        pi0=self.embryo_pi0s[template_embryo],
                        std_dev=self.embryo_sigmas[template_embryo],
                    ) + emission_baf(
                        self.embryo_bafs[k][i],
                        m=0,
                        p=self.pat_haps[0, i],
                        pi0=self.embryo_pi0s[k],
                        std_dev=self.embryo_sigmas[k],
                    )
                else:
                    # Same allele is transmitted
                    z_same[0] = emission_baf(
                        self.embryo_bafs[template_embryo][i],
                        m=self.mat_haps[0, i],
                        p=0,
                        pi0=self.embryo_pi0s[template_embryo],
                        std_dev=self.embryo_sigmas[template_embryo],
                    ) + emission_baf(
                        self.embryo_bafs[k][i],
                        m=self.mat_haps[0, i],
                        p=0,
                        pi0=self.embryo_pi0s[k],
                        std_dev=self.embryo_sigmas[k],
                    )
                    z_same[1] = emission_baf(
                        self.embryo_bafs[template_embryo][i],
                        m=self.mat_haps[0, i],
                        p=1,
                        pi0=self.embryo_pi0s[template_embryo],
                        std_dev=self.embryo_sigmas[template_embryo],
                    ) + emission_baf(
                        self.embryo_bafs[k][i],
                        m=self.mat_haps[0, i],
                        p=1,
                        pi0=self.embryo_pi0s[k],
                        std_dev=self.embryo_sigmas[k],
                    )
                    # Same allele is not transmitted ...
                    z_diff[0] = emission_baf(
                        self.embryo_bafs[template_embryo][i],
                        m=self.mat_haps[0, i],
                        p=0,
                        pi0=self.embryo_pi0s[template_embryo],
                        std_dev=self.embryo_sigmas[template_embryo],
                    ) + emission_baf(
                        self.embryo_bafs[k][i],
                        m=self.mat_haps[0, i],
                        p=1,
                        pi0=self.embryo_pi0s[k],
                        std_dev=self.embryo_sigmas[k],
                    )
                    z_diff[1] = emission_baf(
                        self.embryo_bafs[template_embryo][i],
                        m=self.mat_haps[0, i],
                        p=1,
                        pi0=self.embryo_pi0s[template_embryo],
                        std_dev=self.embryo_sigmas[template_embryo],
                    ) + emission_baf(
                        self.embryo_bafs[k][i],
                        m=self.mat_haps[0, i],
                        p=0,
                        pi0=self.embryo_pi0s[k],
                        std_dev=self.embryo_sigmas[k],
                    )
                z1s[p] = logsumexp(z_same)
                z2s[p] = logsumexp(z_diff)
            llrs = np.array(z1s) - np.array(z2s)
            llr_z[j, :] = llrs

        # Now we make a rough decision rule here for the likelihood-ratio supporting one or the other class ...
        # 1 indicates copying the same allele, -1 indicates that copying different alleles
        # 0 indicates a "missing" variant
        Z = np.zeros(shape=llr_z.shape)
        Z[llr_z < ll_thresh] = -1
        Z[llr_z > ll_thresh] = 1

        # # For each sibling embryo check its "switch-clusters"
        # isolated_switches = []
        # for i in range(len(non_template_ids)):
        #     # Check the switch cluster for sibling i ....
        #     potential_switches = np.where(Z[i, :-1] != Z[i, 1:])[0]
        #     potential_switches_filt = self.refine_recomb_events(
        #         potential_switches, npad=npad
        #     )
        #     isolated_switches.append(potential_switches_filt)

        isolated_switches = self.identify_switch_intervals(Z, npad=npad)
        # Check the total isolated switches ...
        isolated_switches = np.hstack(isolated_switches)
        switch_idx, cnts = np.unique(isolated_switches, return_counts=True)
        # Choose the potential switches by the majority rule ...
        potential_switches_filt = switch_idx[cnts > (m - 1) / 2].astype(int)
        switch_cnts_filt = cnts[cnts > (m - 1) / 2].astype(int)
        return Z, llr_z, potential_switches_filt, switch_cnts_filt

    def identify_switch_intervals(self, Z, npad=5):
        """Identify sign-change intervals in transmission indicator matrix Z.

        Missing or uncertain sites (Z = 0) are imputed by forward-propagating the
        last non-zero sign before applying :meth:`refine_recomb_events` to each row.

        Parameters
        ----------
        Z : np.ndarray
            n_sib x n_info matrix of transmission indicators
            (+1 same allele, -1 different allele, 0 uncertain).
        npad : int
            Half-width of switch cluster used by :meth:`refine_recomb_events`.

        Returns
        -------
        isolated_switches : list of list
            Per-sibling lists of refined switch indices.

        """
        assert Z.ndim == 2
        assert Z.shape[1] > 0
        assert Z.shape[0] > 1
        assert npad > 1
        nsib = Z.shape[0]
        isolated_switches = []
        for i in range(nsib):
            zs = Z[i, :]
            asign = np.sign(zs)
            sz = asign == 0
            while sz.any():
                asign[sz] = np.roll(asign, 1)[sz]
                sz = asign == 0
            signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
            # NOTE: we don't consider the first signchange index...
            potential_switches = np.where(signchange[1:])[0]
            potential_switches_filt = self.refine_recomb_events(
                potential_switches, npad=npad
            )
            isolated_switches.append(potential_switches_filt)
        return isolated_switches

    def second_refine_recomb(
        self, template_embryo=0, maternal=True, start=None, end=None
    ):
        """Refine a crossover interval using QuadHMM on the bracketing SNP window.

        Uses :class:`QuadHMM` pairwise Viterbi paths within the interval
        [start, end] to narrow the crossover location to the tightest flanking
        heterozygous-site pair that is consistent across all sibling pairs.

        Parameters
        ----------
        template_embryo : int
            Index of the reference embryo; default 0.
        maternal : bool
            If True refine maternal crossovers; default True.
        start : int
            Left SNP index (inclusive) of the coarse interval.
        end : int
            Right SNP index (inclusive) of the coarse interval.

        Returns
        -------
        p1 : float
            Refined left boundary position in base pairs.
        p2 : float
            Refined right boundary position in base pairs.

        """
        assert (start is not None) and (end is not None)
        assert start < self.pos.size
        assert end < self.pos.size
        assert start < end
        embryo_ids = np.arange(len(self.embryo_bafs))
        assert template_embryo < embryo_ids.size
        # Make sure there are intermediate SNPs for refinement if necessary ...
        if end - start > 2:
            quad_hmm = QuadHMM()
            paths = []
            for j in embryo_ids:
                if j != template_embryo:
                    bafs = [
                        self.embryo_bafs[template_embryo][start:end],
                        self.embryo_bafs[j][start:end],
                    ]
                    # Use the sibling embryo paths here ...
                    v_path = quad_hmm.viterbi_path(
                        bafs=bafs,
                        pos=self.pos[start:end],
                        mat_haps=self.mat_haps[:, start:end],
                        pat_haps=self.pat_haps[:, start:end],
                        pi0=(self.embryo_pi0s[template_embryo], self.embryo_pi0s[j]),
                        std_dev=(
                            self.embryo_sigmas[template_embryo],
                            self.embryo_sigmas[j],
                        ),
                    )
                    paths.append(v_path)
            # Identify places where there are switches in the local copying paths?
            # Note that these indexes are within the shortened range now ...
            _, _, mat_rec_dict, pat_rec_dict = quad_hmm.isolate_recomb(
                paths[0], paths[1:], window=5
            )
            # They have to be at the same position across all the siblings ...
            mat_rec = [k for k in mat_rec_dict if mat_rec_dict[k] == len(paths)]
            pat_rec = [k for k in pat_rec_dict if pat_rec_dict[k] == len(paths)]
            if maternal:
                if (len(mat_rec) == 1) and (len(pat_rec) == 0):
                    het_idx = np.where((self.mat_haps.sum(axis=0) == 1))[0]
                    pos_het = self.pos[het_idx]
                    pos = self.pos[start:end][mat_rec[0]]
                    start = self.pos[np.argmax(pos_het > pos)]
                    end = self.pos[np.argmin(pos <= pos_het)]
            else:
                if (len(pat_rec) == 1) and (len(mat_rec) == 0):
                    het_idx = np.where((self.pat_haps.sum(axis=0) == 1))[0]
                    pos_het = self.pos[het_idx]
                    pos = self.pos[start:end][mat_rec[0]]
                    start = self.pos[np.argmax(pos_het > pos)]
                    end = self.pos[np.argmin(pos <= pos_het)]
        return self.pos[start], self.pos[end]

    def finalize_recomb_events(
        self, potential_switches, template_embryo=0, maternal=True
    ):
        """Localise potential crossovers to the tightest flanking informative-marker interval.

        For each entry in ``potential_switches`` calls :meth:`second_refine_recomb`
        on the flanking informative-marker pair to obtain a refined base-pair interval.

        Parameters
        ----------
        potential_switches : list
            List of informative-marker indices at which a crossover is suspected
            (output of :meth:`isolate_recomb_events`).
        template_embryo : int
            Index of the reference embryo; default 0.
        maternal : bool
            If True refine maternal crossovers; default True.

        Returns
        -------
        rec_locations : list of tuple
            List of ``(p1, p2)`` base-pair intervals bounding each crossover;
            empty list if ``potential_switches`` is empty.

        """
        if potential_switches == []:
            return []
        else:
            rec_locations = []
            # Get the locations of the informative markers
            info_idx = np.where(self.informative_markers(maternal=maternal))[0]
            assert info_idx.size > 0
            for p in potential_switches:
                # This is the position of the transition at the current resolution ...
                idx1, idx2 = info_idx[p], info_idx[p + 1]
                assert idx1 <= idx2
                (p1, p2) = self.second_refine_recomb(
                    template_embryo=template_embryo,
                    maternal=maternal,
                    start=idx1,
                    end=idx2,
                )
                rec_locations.append((p1, p2))
            return rec_locations

    def estimate_crossovers(
        self, template_embryo=0, ll_thresh=0, maternal=True, npad=5
    ):
        """Full crossover detection pipeline: isolation, majority-rule filtering, and refinement.

        Calls :meth:`isolate_recomb_events` to identify candidate switch positions,
        applies majority-rule filtering, then passes surviving candidates to
        :meth:`finalize_recomb_events` for QuadHMM-based interval refinement.

        Requires :meth:`add_baf`, :meth:`est_sigma_pi0s` to have been called first.

        Parameters
        ----------
        template_embryo : int
            Index of the reference embryo; default 0.
        ll_thresh : float
            Log-likelihood ratio threshold separating "same" from "different"
            allele transmission; default 0 (LLR sign flip).
        maternal : bool
            If True detect maternal crossovers; default True.
        npad : int
            Half-width of the switch-cluster filter; default 5.

        Returns
        -------
        rec_loc : list of tuple
            Refined base-pair intervals ``(p1, p2)`` bounding each crossover.
        switch_cnts : np.ndarray
            Support counts for each filtered switch.

        Notes
        -----
        To obtain the raw transmission-indicator matrix and LLR values call
        :meth:`isolate_recomb_events` directly.

        """
        assert self.embryo_bafs is not None
        assert self.embryo_pi0s is not None
        assert self.embryo_sigmas is not None
        Z, llr_z, potential_switches, switch_cnts = self.isolate_recomb_events(
            template_embryo=template_embryo,
            maternal=maternal,
            ll_thresh=ll_thresh,
            npad=npad,
        )
        rec_loc = self.finalize_recomb_events(
            potential_switches, template_embryo=template_embryo, maternal=maternal
        )
        # We should have the same number of recombination events ...
        assert len(rec_loc) == potential_switches.size
        return rec_loc, switch_cnts

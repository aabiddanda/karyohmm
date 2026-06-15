"""CLI for mosaic cell-fraction estimation using MosaicEst."""

import gzip as gz
import logging

import numpy as np
import polars as pl
import rich_click as click
from scipy.stats import chi2

from karyohmm import DataReader, MosaicEst

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)


@click.command()
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True),
    help="Input data file (NPZ or TSV/CSV) with BAF, LRR, and parental haplotypes.",
)
@click.option(
    "--chrom",
    "-c",
    multiple=True,
    default=(),
    type=str,
    help=(
        "Restrict analysis to these chromosome(s). "
        "Repeatable: -c chr21 -c chr18. "
        "Default: all chromosomes in input."
    ),
)
@click.option(
    "--std-dev-baf",
    required=False,
    default=0.1,
    type=float,
    show_default=True,
    help="BAF emission noise standard deviation at expected-het sites.",
)
@click.option(
    "--switch-err",
    required=False,
    default=0.01,
    type=float,
    show_default=True,
    help="Within-type maternal<->paternal origin-switch probability.",
)
@click.option(
    "--t-rate",
    required=False,
    default=1e-4,
    type=float,
    show_default=True,
    help="Neutral<->aneuploid transition probability per site.",
)
@click.option(
    "--alpha",
    required=False,
    default=0.05,
    type=float,
    show_default=True,
    help="LRT significance threshold for flagging mosaicism.",
)
@click.option(
    "--min-het",
    required=False,
    default=10,
    type=int,
    show_default=True,
    help="Minimum expected-het sites required; chromosomes below this are skipped.",
)
@click.option(
    "--no-lrr",
    is_flag=True,
    default=False,
    help="Ignore LRR data even when present in input (BAF-only mode).",
)
@click.option(
    "--gammas",
    is_flag=True,
    default=False,
    help="Write per-site log forward-variable file for each state.",
)
@click.option(
    "--gzip",
    "-g",
    is_flag=True,
    default=False,
    help="Gzip all output files.",
)
@click.option(
    "--out",
    "-o",
    required=True,
    type=str,
    default="karyohmm_mosaic",
    help="Output file prefix.",
)
def main(
    input,
    chrom,
    std_dev_baf,
    switch_err,
    t_rate,
    alpha,
    min_het,
    no_lrr,
    gammas,
    gzip,
    out,
):
    """Estimate mosaic cell fraction and parental origin per chromosome.

    Runs a 5-state joint BAF + LRR HMM (MosaicEst) on each chromosome and
    reports the MLE cell fraction, 95% CI, inferred parental origin, and a
    likelihood-ratio test for mosaicism.

    States: neutral | maternal-gain | paternal-gain | maternal-loss | paternal-loss
    """
    logging.info(f"Reading input data from {input} ...")
    data_reader = DataReader(mode="Meta")
    data_df = data_reader.read_data(input)
    logging.info("Finished reading input.")

    has_lrr = ("lrr" in data_df.columns) and ("sigmas" in data_df.columns) and (not no_lrr)
    if not has_lrr:
        logging.warning(
            "LRR/sigmas columns not found (or --no-lrr set). "
            "Running in BAF-only mode — detection power will be reduced."
        )

    target_chroms = list(chrom) if chrom else data_df["chrom"].unique().sort().to_list()
    logging.info(f"Analysing {len(target_chroms)} chromosome(s): {', '.join(map(str, target_chroms))}")

    summary_rows = []
    gamma_dfs = []

    for c in target_chroms:
        logging.info(f"[{c}] Starting mosaic estimation ...")
        cur_df = data_df.filter(pl.col("chrom") == c).sort("pos")

        if cur_df.is_empty():
            logging.warning(f"[{c}] No data found — skipping.")
            continue

        mat_haps = np.vstack([cur_df["mat_hap0"].to_numpy(), cur_df["mat_hap1"].to_numpy()])
        pat_haps = np.vstack([cur_df["pat_hap0"].to_numpy(), cur_df["pat_hap1"].to_numpy()])
        bafs = cur_df["baf"].to_numpy()
        pos = cur_df["pos"].to_numpy()
        lrrs = cur_df["lrr"].to_numpy() if has_lrr else None
        sigmas = cur_df["sigmas"].to_numpy() if has_lrr else None

        try:
            m_est = MosaicEst(
                mat_haps=mat_haps,
                pat_haps=pat_haps,
                bafs=bafs,
                pos=pos,
                lrrs=lrrs,
                sigmas=sigmas,
                switch_err=switch_err,
                t_rate=t_rate,
            )
        except ValueError as e:
            logging.warning(f"[{c}] Skipping — {e}")
            continue

        if m_est.n_het < min_het:
            logging.warning(
                f"[{c}] Only {m_est.n_het} expected-het sites (< --min-het {min_het}) — skipping."
            )
            continue

        logging.info(f"[{c}] Phasing complete. {m_est.n_het} expected-het sites, {pos.size} total sites.")
        logging.info(f"[{c}] Running MLE optimisation ...")
        m_est.est_mle_cf(std_dev_baf=std_dev_baf)

        if np.isnan(m_est.mle_cf):
            logging.warning(f"[{c}] MLE optimisation failed — recording NaN.")
            summary_rows.append({
                "chrom": c,
                "n_sites": pos.size,
                "n_het": m_est.n_het,
                "mle_cf": float("nan"),
                "cf_lower": float("nan"),
                "cf_upper": float("nan"),
                "origin": "unknown",
                "lrt": float("nan"),
                "lrt_pval": float("nan"),
                "significant": False,
                "std_dev_baf": std_dev_baf,
            })
            continue

        ci = m_est.ci_mle_cf(std_dev_baf=std_dev_baf)
        lrt_stat = m_est.lrt_cf(std_dev_baf=std_dev_baf)
        lrt_pval = float(chi2.sf(lrt_stat, df=1))
        origin = m_est.infer_origin(std_dev_baf=std_dev_baf)
        significant = lrt_pval < alpha

        logging.info(
            f"[{c}] mle_cf={m_est.mle_cf:.4f} "
            f"CI=[{ci[0]:.4f}, {ci[2]:.4f}] "
            f"origin={origin} "
            f"LRT={lrt_stat:.2f} p={lrt_pval:.2e} "
            f"{'*significant*' if significant else 'not significant'}"
        )

        summary_rows.append({
            "chrom": c,
            "n_sites": pos.size,
            "n_het": m_est.n_het,
            "mle_cf": m_est.mle_cf,
            "cf_lower": ci[0],
            "cf_upper": ci[2],
            "origin": origin,
            "lrt": lrt_stat,
            "lrt_pval": lrt_pval,
            "significant": significant,
            "std_dev_baf": std_dev_baf,
        })

        if gammas:
            alphas, _, _ = m_est.forward_algo_full(cf=m_est.mle_cf, std_dev_baf=std_dev_baf)
            gamma_df = pl.DataFrame({
                state: alphas[k, :]
                for k, state in enumerate(MosaicEst.STATE_NAMES)
            }).with_columns(
                pl.lit(c).alias("chrom"),
                pl.Series("pos", pos),
                pl.lit(m_est.mle_cf).alias("mle_cf"),
            )
            cols_first = ["chrom", "pos", "mle_cf"]
            gamma_df = gamma_df.select(
                cols_first + [col for col in gamma_df.columns if col not in cols_first]
            )
            gamma_dfs.append(gamma_df)

    if not summary_rows:
        logging.warning("No chromosomes were successfully analysed. No output written.")
        return

    # Write summary
    summary_df = pl.DataFrame(summary_rows)
    ext = ".tsv.gz" if gzip else ".tsv"
    summary_fp = f"{out}.mosaic.summary{ext}"
    if gzip:
        with gz.open(summary_fp, "wb") as f:
            summary_df.write_csv(f, separator="\t")
    else:
        summary_df.write_csv(summary_fp, separator="\t")
    logging.info(f"Wrote per-chromosome mosaic summary to {summary_fp}")

    # Write per-site gammas if requested
    if gammas and gamma_dfs:
        gamma_fp = f"{out}.mosaic.gammas{ext}"
        all_gammas = pl.concat(gamma_dfs)
        if gzip:
            with gz.open(gamma_fp, "wb") as f:
                all_gammas.write_csv(f, separator="\t")
        else:
            all_gammas.write_csv(gamma_fp, separator="\t")
        logging.info(f"Wrote per-site forward-variable log-probabilities to {gamma_fp}")

    logging.info("Mosaic estimation complete.")

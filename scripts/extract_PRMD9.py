from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import numpy as np
from pybedtools import BedTool
from pwm_density import PWMExtractor
import pandas as pd
import polars as pl
from scipy.stats import ks_2samp
from gff_utils import Expander

def bootstrap(intersect_df: pd.DataFrame, window_size: int = 2_000, N: int = 1_000, alpha: float = 0.05):
    bootstrapped_samples = []
    if isinstance(intersect_df, pl.DataFrame):
        intersect_df = intersect_df.to_pandas()
    extractor = PWMExtractor()
    for _ in tqdm(range(N)):
        sample_df = intersect_df.sample(frac=1.0, replace=True)
        density = extractor.extract_density(pl.from_pandas(sample_df), window_size=window_size)
        density = density / np.mean(density)
        bootstrapped_samples.append(density)
    bootstrapped_samples = pd.DataFrame(bootstrapped_samples)
    
    # global density
    global_density = extractor.extract_density(pl.from_pandas(intersect_df), window_size=window_size)
    global_density = global_density / np.mean(global_density)

    # aggregate statistics
    inter_mean = pd.Series(global_density).round(3).to_frame(name='average')
    lower_ci = bootstrapped_samples.quantile(alpha/2).round(3).to_frame(name='q0025')
    upper_ci = bootstrapped_samples.quantile(1-alpha/2).round(3).to_frame(name='q0975')
    return pd.concat([
                        inter_mean, 
                        lower_ci, 
                        upper_ci
                    ], axis=1)


def load_PRMD9_hotspots(PRMD9_PATH):
    hotspots = {f.name.split('.')[0]: pl.read_csv(f, 
                                              separator="\t", 
                                              has_header=False, 
                                              new_columns=["seqID", "start", "end"]
                                             ) for f in PRMD9_PATH.glob("*peaks.hs1.bed")}
    return hotspots
    
def extract_PRMD9_densities(PRMD9_PATH, g4_bed, g4_controls_bed, allele, outdir, window_size=2000, N=1000, alpha=0.05):
    PRMD9_PATH = Path(PRMD9_PATH).resolve()
    hotspots = load_PRMD9_hotspots(PRMD9_PATH)
    df = hotspots[allele]
    e = Expander(window_size=window_size)

    df = df.with_columns(pl.lit("+").alias("strand"))
    prmd9_expanded = e.expand_windows(df, loci="mid")
    df_bed = BedTool.from_dataframe(prmd9_expanded.to_pandas()).sort()

    # G4 Hunter
    intersect_df = pl.read_csv(
                df_bed.intersect(g4_bed, wo=True).fn,
                has_header=False,
                separator="\t",
                new_columns=["seqID", "start", "end", "strand", "mid", "chromosome", "motif_start", "motif_end", "overlap"]
    )
    bootstrapped_density = bootstrap(intersect_df, window_size=window_size, N=N, alpha=alpha)

    # Control
    bootstrapped_density.to_csv(
                outdir.joinpath(f"PRMD9_allele_{allele}_g4hunter_density_bootstrap_{N}_window_{window_size}_alpha_{alpha}.csv"),
                            sep=",", 
                            mode="w")

    intersect_df = pl.read_csv(
                df_bed.intersect(g4_controls_bed, wo=True).fn,
                has_header=False,
                separator="\t",
                new_columns=["seqID", "start", "end", "strand", "mid", "chromosome", "motif_start", "motif_end", "overlap"]
    )
    bootstrapped_control_density = bootstrap(intersect_df, window_size=window_size, N=N, alpha=alpha)

    bootstrapped_control_density.to_csv(
                outdir.joinpath(f"PRMD9_allele_{allele}_g4hunter_controls_density_bootstrap_{N}_window_{window_size}_alpha_{alpha}.csv"),
                            sep=",", 
                            mode="w")

if __name__ == "__main__":

    import argparse
    mutations = ["smallins", "smalldel", "mnp", "ins", "del", "snp"]
    parser = argparse.ArgumentParser()
    parser.add_argument("allele", type=str)
    parser.add_argument("--extraction", type=str, default="/storage/group/izg5139/default/nicole/MirrorRTR/g4_results/chm13v2_g4hunter.txt")
    parser.add_argument("--control", type=str, default="/storage/group/izg5139/default/nicole/g4_t2t_analysis/chm13v2_g4hunter.controls.txt")
    parser.add_argument("--PRMD9_PATH", type=str, default="../DSBhotspots_hs1")
    parser.add_argument("--outdir", type=str, default="densities_bootstrap")
    parser.add_argument("--window_size", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05) # 95% confidence interval
    parser.add_argument("--N", type=int, default=1000) # 1000 iterations in bootstrap
    args = parser.parse_args()
    
    extraction = args.extraction
    window_size = args.window_size

    PRMD9_PATH = Path(args.PRMD9_PATH).resolve()
    outdir = Path(args.outdir).resolve()
    allele = args.allele
    outdir.mkdir(exist_ok=True)
    extractor = PWMExtractor()

    # g4 bed
    g4_df = pl.read_csv(args.extraction, separator="\t")
    g4hunter_bed = BedTool.from_dataframe(g4_df.select(["seqID", "start", "end"]).to_pandas()).sort()

    # controls
    g4_control_df = pl.read_csv(args.control, separator="\t")
    g4_controls_bed = BedTool.from_dataframe(g4_control_df.select(["seqID", "start", "end"]).to_pandas()).sort()
    # intersections
    ######################### G4 HUNTER DENSITY #################################################################3
    extract_PRMD9_densities(PRMD9_PATH, 
                            g4hunter_bed, 
                            g4_controls_bed, 
                            allele=allele, 
                            outdir=outdir, 
                            window_size=window_size, 
                            N=args.N, 
                            alpha=args.alpha)

import os
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from pybedtools import BedTool
# from constants import ConfigPaths
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import percentileofscore
from statsmodels.stats.multitest import multipletests
from dotenv import load_dotenv

def load_sequence_sizes() -> pl.DataFrame:
    sequence_sizes = pl.read_csv(Path("~/biolab/t2t_g4/centromeric/genome.txt").expanduser(),
                                 # ConfigPaths.SEQUENCE_SIZES.value,
                                 separator="\t",
                                 has_header=False,
                                 new_columns=["seqID", "size"]
                                 )
    return sequence_sizes

def map_to_bins(df: pl.DataFrame, total_bins: int) -> pl.DataFrame:
    if "size" not in df.columns:
        sequence_sizes = load_sequence_sizes()
        df = df.join(
                    sequence_sizes, 
                     how="left",
                     on="seqID"
                     )
        
    df_assigned = df.with_columns(
                    start_bin=((pl.col("start") * total_bins / pl.col("size")).floor() + 1).cast(pl.Int32),
                    end_bin=(((pl.col("end")-1) * total_bins / pl.col("size")).floor() + 1).cast(pl.Int32)
                    )
    return df_assigned
    
def load_centromere(centro, index=None) -> pl.DataFrame:
    df = pl.read_csv(centro, 
                     separator="\t",
                     columns=range(4),
                     skip_rows=1,
                     new_columns=["seqID", "start", "end", "compartment"]
                     )\
        .with_columns(
                    pl.col("compartment")
                        .map_elements(lambda n: n.split("_")[0], return_dtype=str)
                        .alias("compartment")
                )
    centromeric_annotations = set(df["compartment"])
    centromere_df = []
    for c in centromeric_annotations:
        temp = df.filter(pl.col("compartment") == c).to_pandas()
        temp_df = pl.read_csv(
                        BedTool.from_dataframe(temp)
                                        .sort()
                                        .merge(c="4", o="distinct").fn,
                        has_header=False,
                        separator="\t",
                        new_columns=["seqID", "start", "end", "compartment"]
                        )
        centromere_df.append(temp_df)
    centromere_df = pl.concat(centromere_df).sort(["seqID", "start"])
    return centromere_df

def centromere_to_bins(centromere_df):
    bin_compartments = defaultdict(set)
    for row in centromere_df.iter_rows(named=True):
        start = row['start_bin']
        end = row['end_bin']
        compartment = row['compartment']

        for binn in range(start, end+1):
            bin_compartments[binn].add(compartment)
    return bin_compartments

def create_windows(genome_file: str, total_bins: int = 2000) -> pl.DataFrame:
    windows = BedTool().window_maker(g=genome_file, n=total_bins)
    regions = []
    for window in windows:
        regions.append({
                        "seqID": window.chrom,
                        "start": window.start,
                        "end": window.stop,
                       })
    total_chromosomes = len({binn["seqID"] for binn in regions})
    assert len(regions) == total_bins * total_chromosomes, f"{len(regions)} != {total_bins}"
    regions_df = pl.DataFrame(regions)\
                    .with_columns(
                                pl.int_range(pl.len(), dtype=pl.UInt32).over("seqID").alias("bin")
                            )\
                    .with_columns(
                            bin=pl.col("bin")+1
                    )
    return regions_df

def extract_region_gc_content(regions_df: pl.DataFrame, fasta_file) -> pl.DataFrame:
    regions_bed = BedTool.from_dataframe(regions_df.to_pandas()).sort()
    seqfa = regions_bed.sequence(fi=str(fasta_file),
                                 name=True, 
                                 tab=True)
    regions_df_with_gc = []
    with open(seqfa.seqfn, mode="r", encoding="utf-8") as f:
        for line in f:
            line, sequence = line.strip().split("\t")
            sequence = sequence.lower()
            sequence_length = len(sequence.strip())
            gc_content = sequence.count("g") + sequence.count("c")
            gc_proportion = gc_content / sequence_length
            allele = line.split('::')[0]
            seqID = line.split('::')[1].split(':')[0]
            start, end = line.split(':')[-1].split('-')
            start = int(start)
            end = int(end)
            regions_df_with_gc.append(
                            {
                                "seqID": seqID,
                                "start": start,
                                "end": end,
                                "bin": allele,
                                "gc_proportion": gc_proportion,
                                "gc_content": gc_content,
                                "sequence_length": sequence_length
                            }
                    )
    regions_df_with_gc = pl.DataFrame(regions_df_with_gc)
    return regions_df_with_gc

def calculate_genome_density(motif_bed) -> float:
    sequence_sizes = load_sequence_sizes()
    genome_size = sequence_sizes['size'].sum()
    motif_bp = pl.read_csv(
                                motif_bed
                                .sort()
                                .merge().fn, 
                                has_header=False, 
                                separator="\t", 
                                new_columns=["seqID", "start", "end"]
                            )\
                            .with_columns(
                                    length=pl.col("end")-pl.col("start")
                            )["length"].sum()
    genome_density = 1e6 * motif_bp / genome_size
    print(f"Total density: {genome_density:.2f}.")
    return genome_density

def calculate_region_motif_coverage(region_df, motif_bed) -> pl.DataFrame:
    genome_density = calculate_genome_density(motif_bed)
    region_bed = BedTool.from_dataframe(region_df.to_pandas()).sort()
    COVERAGE_FIELDS = ["totalHits", "overlappingBp", "compartmentLength", "coverage"]
    print("MORE WORK? MOTIF WORK!")
    region_df = pl.read_csv(
                    region_bed.coverage(motif_bed).fn,
                    has_header=False,
                    separator="\t",
                    new_columns=region_df.columns + COVERAGE_FIELDS
                ).with_columns(
                            (pl.col("coverage") * 1e6).alias("motif_density"),
                )\
                .with_columns(
                            (pl.col("motif_density") / genome_density).alias("fold_enrichment")
                )
    print("JOBS DONE!")
    return region_df

def construct_regions_pipe(genome_file: str, fasta_file: str, g4_file: str, total_bins: int) -> pl.DataFrame:
    regions_df = create_windows(genome_file, total_bins)
    region_df_gc = extract_region_gc_content(regions_df, fasta_file)
    motif_df = pl.read_csv(g4_file, separator="\t").drop(["NBR"])
    motif_bed = BedTool.from_dataframe(motif_df.to_pandas()).sort()
    region_df_gc = calculate_region_motif_coverage(region_df_gc, motif_bed)
    return region_df_gc

def map_compartments_to_bins_agnostic(centromere_df, 
                             bin_sizes: dict[str, int],
                             remove_zero: bool = False) -> dict[str, pd.DataFrame]:
    chromosomes = centromere_df['seqID'].unique()
    # chromosomes = [seqID for seqID in bin_sizes]
    bin_categories = dict()
    compartments = ["bsat", 
                    "gsat", 
                    "censat", 
                    "ct", 
                    "hsat1A", 
                    "hsat1B", 
                    "hsat2", 
                    "hsat3", 
                    "dhor", 
                    "hor", 
                    "mon",
                    "rDNA"]
    valid = set(compartments)
    for chromosome in chromosomes:
        bin_categories[chromosome] = {i: set() for i in range(1, bin_sizes[chromosome]+1)}
    for row in tqdm(centromere_df.iter_rows(named=True)):
        start = row['start_bin']
        end = row['end_bin']
        chromosome = row['seqID']
        compartment = row['compartment']
        if compartment not in valid or chromosome == 'chrM':
            continue
        
        start = int(start)
        end = int(end)
        for i in range(start, end+1):
            bin_categories[chromosome][i].add(compartment)     

    new_bins = {}
    compartments = list(compartments)
    for chromosome in chromosomes:
        if chromosome not in bin_categories:
            continue
        temp = bin_categories[chromosome]
        new_bins.update({chromosome: []})
        total_bins = bin_sizes[chromosome]
        for i in range(1, total_bins+1):
            comps = temp[i]
            new_bins[chromosome].append([])
            for c in compartments:
                if c in comps:
                    new_bins[chromosome][-1].append(1)
                else:
                    new_bins[chromosome][-1].append(0)
        new_bins[chromosome] = pd.DataFrame(new_bins[chromosome], columns=compartments)

        if remove_zero:
            for col in new_bins[chromosome]:
                if new_bins[chromosome][col].sum() == 0:
                    new_bins[chromosome].drop(columns=[col], inplace=True)
    return new_bins


def map_compartments_to_bins(centromere_df, 
                             total_bins: int = 2000,
                             remove_zero: bool = False):
    chromosomes = centromere_df['seqID'].unique()
    bin_categories = defaultdict(lambda : {i: set() for i in range(1, total_bins+1)})
    compartments = ["bsat", 
                    "gsat", 
                    "censat", 
                    "ct", 
                    "hsat1A", 
                    "hsat1B", 
                    "hsat2", 
                    "hsat3", 
                    "dhor", 
                    "hor", 
                    "mon",
                    "rDNA"]
    valid = set(compartments)
    for row in centromere_df.iter_rows(named=True):
        start = row['start_bin']
        end = row['end_bin']
        chromosome = row['seqID']
        compartment = row['compartment']
        if compartment not in valid or chromosome == 'chrM':
            continue
        
        start = int(start)
        end = int(end)
        for i in range(start, end+1):
            bin_categories[chromosome][i].add(compartment)   

#    intervals = {}
#    for chromosome in bin_categories:
#        intervals.update({chromosome: defaultdict(set)})
#        temp = bin_categories[chromosome]
#        positions = sorted(list(temp.keys()))
#        current_list = None
#        least_bin = None
#        for i in range(len(positions)-1):
#            if current_list is None:
#                least_bin = positions[i]
#                current_list = temp[positions[i]]
#            if temp[positions[i+1]] != current_list:
#                intervals[chromosome].update({(least_bin, positions[i]): current_list})
#                current_list = None  

    new_bins = {}
    compartments = list(compartments)
    for chromosome in chromosomes:
        temp = bin_categories[chromosome]
        new_bins.update({chromosome: []})
        for i in range(1, total_bins+1):
            comps = temp[i]
            new_bins[chromosome].append([])
            for c in compartments:
                if c in comps:
                    new_bins[chromosome][-1].append(1)
                else:
                    new_bins[chromosome][-1].append(0)
        new_bins[chromosome] = pd.DataFrame(new_bins[chromosome], columns=compartments)

        if remove_zero:
            for col in new_bins[chromosome]:
                if new_bins[chromosome][col].sum() == 0:
                    new_bins[chromosome].drop(columns=[col], inplace=True)
    return new_bins


def calculate_bin_counts(df: pl.DataFrame, total_bins: int = 2_000) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    bin_counts = defaultdict(lambda: np.zeros(total_bins))
    df_merged = map_to_bins(df, total_bins=total_bins)
    
    df_merged_grouped = df_merged.group_by(["seqID", "start_bin", "end_bin"], 
                                           maintain_order=True)\
                                    .agg(totalCounts=pl.col("start").count())
    chromosomes = df_merged_grouped['seqID'].unique()
    print(f"Total chromosomes {len(chromosomes)}.")
    
    for chromosome in chromosomes:
        for i in range(total_bins):
            bin_counts[chromosome][i] = 0 
            
    # calculate counts per bin
    for row in tqdm(df_merged_grouped.iter_rows(named=True), total=df_merged_grouped.shape[0]):
        counts = row['totalCounts']
        start = row['start_bin']
        end = row['end_bin']
        chromosome = row['seqID']
        for i in range(start, end+1):
            bin_counts[chromosome][i-1] += counts  
    
    # calculate fold enrichment
    for chromosome in chromosomes:
        bin_counts[chromosome] = pd.DataFrame(bin_counts[chromosome], columns=["counts"])
        bin_counts[chromosome].index = bin_counts[chromosome].index.map(lambda x: x+1)
        bin_counts[chromosome]["enrichment"] = bin_counts[chromosome]["counts"] / np.mean(bin_counts[chromosome]["counts"])
    return bin_counts

def perform_test(observed_res, residuals):
    percentile = percentileofscore(residuals, observed_res, kind='rank')
    if observed_res > np.median(residuals):
        p_value = 1 - (percentile / 100)
    else:
        p_value = percentile / 100
    return percentile, p_value

def evaluate_stars(pval: float) -> str:
    if pval < 0.0001:
        return "*" * 4
    if pval < 0.001:
        return "*" * 3
    if pval < 0.01:
        return "*" * 2
    if pval < 0.05:
        return "*"
    return "ns"

def adjust_gc(df: pd.DataFrame, model_dir: str, chunk_size: int, degree: int) -> pl.DataFrame:
               
    '''
    Input: 
    df: dataframe to perform GC adjustment on Fold Enrichment
    -------------------------
    - Loads model
    - Performs GC adjustment
    -------------------------
    Returns: GC-adjusted dataframe for motif density
    '''
    model_dir = Path(model_dir).resolve()
    assert model_dir.is_dir()

    def _load_model(model_dir: str, chunk_size: int, degree: int):
        import joblib
        model_path = Path(model_dir).resolve().joinpath("models", f"linreg_model_degree_{degree}_CHUNK_{chunk_size}.pkl")
        with open(model_path, 'rb') as f:
            return joblib.load(f)

    def _load_residuals(model_dir: str, chunk_size: int, degree: int, bias: bool = True):
        residuals_path = model_dir.joinpath("residuals", f"residuals_chunk_{chunk_size}_degree_{degree}_bias_{bias}.txt")
        residuals = []
        with open(residuals_path, mode="r", encoding="utf-8") as f:
            for line in f:
                residuals.append(float(line.strip()))
        return residuals

    loaded_pipe = _load_model(model_dir, chunk_size, degree=degree)
    residuals = _load_residuals(model_dir, chunk_size, degree)

    if "gc_proportion" not in df:
        raise  KeyError(f"GC-Content has not been calculated.")
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    X_test = df[["gc_proportion"]]
    y_pred = pl.Series(loaded_pipe.predict(X_test)).to_frame("predicted_enrichment")
    df_enrichment = pl.concat([
                                        pl.from_pandas(df), 
                                        y_pred
                                    ], how="horizontal")\
                                .with_columns(
                                            (pl.col("fold_enrichment") - pl.col("predicted_enrichment")).alias("res")
                                )\
                                .with_columns(
                                            pl.col("res")
                                            .map_elements(lambda res: perform_test(res, residuals),
                                                         ).alias("test_statistic")
                                )\
                                .with_columns(
                                        pl.col("test_statistic").list.get(0).alias("percentile"),
                                        pl.col("test_statistic").list.get(1).alias("pval")
                                    )
    p_values = list(df_enrichment["pval"])
    corrected_pvals = multipletests(p_values, method='fdr_bh')[1]
    df_enrichment = pl.concat([
                                df_enrichment,
                                pl.Series(corrected_pvals).to_frame("adj_pval"),
                            ], how="horizontal"
                            )\
                        .with_columns(
                                pl.col("pval").map_elements(evaluate_stars, return_dtype=str).alias("significance")
                        )
    return df_enrichment 


if __name__ == "__main__":
    
    total_bins = 2000
    centromere_df = map_to_bins(centromere_df, total_bins=total_bins)
    centro = os.getenv("CENTROMERE")
    df = load_centromere(centro)
    breakpoint()

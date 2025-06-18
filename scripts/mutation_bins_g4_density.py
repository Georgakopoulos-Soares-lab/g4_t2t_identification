import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
from pybedtools import BedTool
import pybedtools
from constants import ConfigPaths
from gff_utils import Expander, CoverageExtractor
from scipy.stats import chi2_contingency
import sys

pybedtools.set_tempdir("/scratch/nmc6088")
SAME_COLUMNS = ["seqID", "start", "end", "strand", "gc_proportion", "gc_content", "sequence_length"]

def contigency(row) -> float:
    """
    Performs a chi-squared contingency test for a given row of overlap/control counts.

    Args:
        row (dict): Dictionary with keys 'overlappingBp', 'not_g4', 'overlappingBp_control', 'not_control'.

    Returns:
        float: p-value from the chi-squared test.
    """
    array = np.array([
                      [row['overlappingBp'], row['not_g4']],
                      [row['overlappingBp_control'], row['not_control']]
                    ], dtype=np.int64)
    return chi2_contingency(array).pvalue

def extract_gc_content(fasta_sequence) -> pl.DataFrame:
    """
    Extracts GC content and related statistics from a fasta sequence file.

    Args:
        fasta_sequence: BedTool sequence object with .seqfn attribute (tab-separated sequence file).

    Returns:
        pl.DataFrame: DataFrame with columns for seqID, start, end, strand, gc_proportion, gc_content, sequence_length.
    """
    df_with_gc = []
    with open(fasta_sequence.seqfn, mode="r", encoding="utf-8") as f:
        for line in f:
            line, sequence = line.strip().split("\t")
            sequence = sequence.lower()
            sequence_length = len(sequence.strip())
            gc_content = sequence.count("g") + sequence.count("c")
            gc_proportion = gc_content / sequence_length
            strand = line.split('::')[0]
            seqID = line.split('::')[1].split(':')[0]
            start, end = line.split(':')[-1].split('-')
            start = int(start)
            end = int(end)
            df_with_gc.append(
                        {
                            "seqID": seqID,
                             "start": start,
                             "end": end,
                             "strand": strand,
                             "gc_proportion": gc_proportion,
                             "gc_content": gc_content,
                             "sequence_length": sequence_length
                        }
                    )
    df_with_gc = pl.DataFrame(df_with_gc)
    return df_with_gc

def main(window_size: int, mutation_type: str, destdir: str):
    """
    Main workflow for extracting mutation-centered windows, calculating GC content, and motif coverage.

    Args:
        window_size (int): Size of the window around each mutation.
        mutation_type (str): Type of mutation to analyze (e.g., 'ins', 'del').
        destdir (str): Output directory for results.

    Returns:
        None
    """
    destdir = Path(destdir).resolve()
    destdir.mkdir(exist_ok=True)
    
    # load dataframes
    g4_df = pd.read_table(ConfigPaths.G4HUNTER.value).drop(columns=['NBR'])
    g4_control_df = pd.read_table(ConfigPaths.CONTROL_G4HUNTER.value)

    g4_bed = BedTool.from_dataframe(g4_df).sort()
    g4_control_bed = BedTool.from_dataframe(g4_control_df).sort()

    # load chromosome sizes     
    human_chromosomes = pl.read_csv(ConfigPaths.GENOME_SIZE.value, 
                                    has_header=False, 
                                    separator="\t", 
                                    new_columns=["seqID", "size"]
                               )

    # load mutations
    mutations = {mutation.name.split('_')[-1].split('.')[0]: mutation for mutation in ConfigPaths.PROCESSED_MUTATIONS.value.glob("*.bed")}
    print(mutations)

    print(f"Total processed mutations `{len(mutations)}`.")
    e = Expander(window_size=window_size)
    
    # start
    mutations_df = pl.read_csv(
                            mutations[mutation_type],
                            has_header=False, 
                            separator="\t",
                            new_columns=["seqID", "start", "end", "count"]
                ).with_columns(pl.lit("+").alias("strand")).drop(['count'])
    # filter out chrY
    mutations_df = mutations_df.filter(pl.col("seqID") != "chrY")
    
    mutations_df_expanded = e.expand_windows(mutations_df, loci="mid")\
                                    .join(human_chromosomes, how="left", on="seqID")\
                                    .with_columns(
                                            pl.min_horizontal(pl.col("end"), pl.col("size")).alias("end")
                                    )
    
    # fetch gc content of the expanded regions
    mutations_bed_exp = BedTool.from_dataframe(mutations_df_expanded.to_pandas()).sort()
    mutations_bed_exp_gc = mutations_bed_exp.sequence(fi=str(ConfigPaths.FASTA.value), 
                                                      tab=True, 
                                                      name=True)
    mutations_df_expanded_gc = extract_gc_content(mutations_bed_exp_gc)
    mutations_bed_exp_gc = BedTool.from_dataframe(mutations_df_expanded_gc.to_pandas()).sort()

    # extract coverage of the region with G4 motifs and Control motifs to assess enrichment
    mut_g4_coverage = pl.read_csv(
                    mutations_bed_exp_gc.coverage(g4_bed).fn,
                    has_header=False,
                    separator="\t",
                    new_columns=mutations_df_expanded_gc.columns + CoverageExtractor.COVERAGE_FIELDS
        )

    mut_control_g4_coverage = pl.read_csv(
                    mutations_bed_exp_gc.coverage(g4_control_bed).fn,
                    has_header=False,
                    separator="\t",
                    new_columns=mutations_df_expanded_gc.columns + CoverageExtractor.COVERAGE_FIELDS
            )

    mut_g4_coverage = mut_g4_coverage.join(
                                mut_control_g4_coverage,
                                how="left",
                                on=SAME_COLUMNS,
                                suffix="_control"
    )\
            .with_columns(
                    window=pl.lit(window_size)
            )\
        .with_columns(pl.lit(mutation_type).alias("mutation_type"))
    
    dest_file = Path(f"{destdir}/mutation_{mutation_type}_window_{window_size}_g4_and_control_coverage.csv")
    mut_g4_coverage.write_csv(dest_file, separator=",")

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="""""")
    parser.add_argument("--window_size", type=int, default=250)
    parser.add_argument("--mutation_type", type=str, default="ins")
    parser.add_argument("--destdir", type=str, default="../mutation_data")
    args = parser.parse_args()

    main(window_size=args.window_size, 
         mutation_type=args.mutation_type,
         destdir=args.destdir,
         )

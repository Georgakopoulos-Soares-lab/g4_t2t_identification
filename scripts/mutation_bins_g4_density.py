import polars as pl
import numpy as np
import pandas as pd
from pathlib import Path
from pybedtools import BedTool
import pybedtools
from constants import ConfigPaths
from gff_utils import Expander, CoverageExtractor
from adjust_gc_content import adjust_for_gc_content
from scipy.stats import chi2_contingency
import sys

pybedtools.set_tempdir("/scratch/nmc6088")
SAME_COLUMNS = ["seqID", "start", "end", "strand", "sequence", "gc_proportion",	"gc_content", "sequence_length"]

def contigency(row):
    array = np.array([
                      [row['overlappingBp'], row['not_g4']],
                      [row['overlappingBp_control'], row['not_control']]
                    ], dtype=np.int64)
    return chi2_contingency(array).pvalue

def extract_gc_content(fasta_sequence):
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
                             "sequence": sequence,
                             "gc_proportion": gc_proportion,
                             "gc_content": gc_content,
                             "sequence_length": sequence_length
                        }
                    )
    df_with_gc = pl.DataFrame(df_with_gc)
    return df_with_gc

def main(window_size: int, mutation_type: str, mutations_path: str, destdir: str):
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
    mutations = {mutation.name.split('_')[1].split('.')[0]: mutation for mutation in Path(mutations_path).glob("*.bed")}
    e = Expander(window_size=window_size)
    
    # start
    mutations_df = pl.read_csv(
                            mutations[mutation_type],
                            has_header=False, 
                            separator="\t",
                            new_columns=["seqID", "start", "end", "count"]
                ).with_columns(pl.lit("+").alias("strand")).drop(['count'])
    mutations_df = mutations_df.filter(pl.col("seqID") != "chrY") 
    
    mutations_df_expanded = e.expand_windows(mutations_df, loci="mid")\
                                    .join(human_chromosomes, how="left", on="seqID")\
                                    .with_columns(
                                            pl.min_horizontal(pl.col("end"), pl.col("size")).alias("end")
                                    )
    
    mutations_bed_exp = BedTool.from_dataframe(mutations_df_expanded.to_pandas()).sort()
    mutations_bed_exp_gc = mutations_bed_exp.sequence(fi=ConfigPaths.FASTA.value, 
                                                      tab=True, 
                                                      name=True)
    mutations_df_expanded_gc = extract_gc_content(mutations_bed_exp_gc)

    mutations_bed_exp_gc = BedTool.from_dataframe(mutations_df_expanded_gc.to_pandas()).sort()
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

def adjust_gc_content(mut_df: pl.DataFrame, g4_gw_density: float, control_gw_density: float):
    mut_g4_coverage_df_grped = mut_df.group_by(["mutation_type", "window"])\
                                                .agg(
                                                        pl.col("overlappingBp").sum(),
                                                        pl.col("compartmentLength").sum().alias("mutation_area_len"),
                                                        pl.col("overlappingBp_control").sum(),
                                                        pl.col("gc_content").sum()
                                                )\
                                                .with_columns(
                                                        (pl.col("overlappingBp") * 1e6 / pl.col("mutation_area_len")).alias("density"),
                                                        (pl.col("overlappingBp_control") * 1e6 / pl.col("mutation_area_len")).alias("density_control"),
                                                        (pl.col("gc_content") / pl.col("mutation_area_len")).alias("gc_proportion")
                                                    )\
                                                .with_columns(
                                                            (pl.col("density") / g4_gw_density).alias("fold_enrichment"),
                                                            (pl.col("density_control") / control_gw_density).alias("fold_enrichment_control")
                                                )\
                                                .with_columns(
                                                            (pl.col("fold_enrichment") / pl.col("fold_enrichment_control")).alias("fe_enrichment")
                                                )\
                                                .with_columns(
                                                            (pl.col("mutation_area_len") - pl.col("overlappingBp")).alias("not_g4"),
                                                            (pl.col("mutation_area_len") - pl.col("overlappingBp_control")).alias("not_control")
                                                        )\
                                                .with_columns(
                                                                pl.struct([pl.col("overlappingBp"), pl.col("overlappingBp_control"), pl.col("not_g4"), pl.col("not_control")])\
                                                                .map_elements(contigency, return_dtype=float).alias("pval")
                                                )\
                                                .with_columns(
                                                                pl.col("pval").map_elements(map_stars, return_dtype=str).alias("sig_stars")
                                                )
    mut_g4_coverage_df_grped = adjust_for_gc_content(mut_g4_coverage_df_grped)
    return mut_g4_coverage_df_grped


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=250)
    parser.add_argument("--mutation_type", type=str, default="ins")
    parser.add_argument("--mutations_path", type=str, default="")
    parser.add_argument("--destdir", type=str, default="")

    args = parser.parse_args()

    main(window_size=args.window_size, 
         mutation_type=args.mutation_type,
         mutations_path=args.mutations_path,
         destdir=args.destdir,
         )


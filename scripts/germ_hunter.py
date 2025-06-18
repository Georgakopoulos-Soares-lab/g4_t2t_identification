import math
import re
from pathlib import Path
from termcolor import colored
import pandas as pd
import polars as pl
from gff_utils import Expander 
from pybedtools import BedTool
import pybedtools
from collections import defaultdict
from tqdm import tqdm
from Bio.Seq import Seq
import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests
from scipy.stats import chi2_contingency

def binom_test_func(row: dict) -> float:
    """
    Performs a binomial test for a given row of mutation and motif counts.

    Args:
        row (dict): Dictionary with keys 'tri_mut_count', 'base_g4_tri_counts', 'mutation_bias'.

    Returns:
        float: p-value from the binomial test.
    """
    return stats.binomtest(row["tri_mut_count"], 
                           row["base_g4_tri_counts"], 
                           row["mutation_bias"], 
                           alternative='two-sided').pvalue

def contigency(row: dict) -> float:
    """
    Performs a chi-squared contingency test for a given row of mutation and motif counts.

    Args:
        row (dict): Dictionary with keys 'tri_mut_count', 'base_tri_counts', 'totalTimesMutated', 'gw_counts'.

    Returns:
        float: p-value from the chi-squared test.
    """
    array = np.array([
                      [row['tri_mut_count'], row['base_tri_counts'] - row['tri_mut_count']],
                      [row['totalTimesMutated'], row['gw_counts'] - row['totalTimesMutated']]
                     ], 
                     dtype=np.int32)
    return chi2_contingency(array).pvalue

def map_stars(pval: float) -> str:
    """
    Maps a p-value to a significance string (stars or 'ns').

    Args:
        pval (float): The p-value to evaluate.

    Returns:
        str: Significance as stars or 'ns'.
    """
    if pval < 0.0001:
        return "****"
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "**"
    if pval < 0.05:
        return "*"
    return "ns"

def merge_motifs(merged_motifs: pl.DataFrame, genome_size: str) -> pl.DataFrame:
    """
    Expands motif regions by one base pair and joins with chromosome sizes.

    Args:
        merged_motifs (pl.DataFrame): DataFrame of merged motif regions.
        genome_size (str): Path to chromosome size file.

    Returns:
        pl.DataFrame: DataFrame with expanded motif regions.
    """
    merged_motifs_df = []
    for row in merged_motifs.iter_rows(named=True):
        seqID = row['seqID']
        start = row['start']
        end = row['end']
        total_length = end - start 

        multiple_starts = list(map(int, row['multiple_starts'].split(",")))
        multiple_sequences = row['multiple_sequences'].split(",")
        merged_sequence = ""

        for sequence, (start_previous, start_current) in zip(multiple_sequences, zip(multiple_starts, multiple_starts[1:])):
            merged_sequence += sequence[:start_current - start_previous]
        merged_sequence += multiple_sequences[-1]
        assert len(merged_sequence) == total_length
        merged_motifs_df.append(
                        {
                                "seqID": seqID,
                                "start": start,
                                "end": end,
                                "sequence": merged_sequence,
                                "length": total_length,
                            }
                    )
    merged_motifs_df = pl.from_pandas(pd.DataFrame(merged_motifs_df))
    
    ## Expand one base pair upstream and downstream to include the trinucleotide at the borders
    chromosome_sizes = pl.read_csv(genome_size,
                                  has_header=False,
                                  separator="\t",
                                  new_columns=["seqID", "size"])
    merged_motifs_df = merged_motifs_df.join(
                                        chromosome_sizes,
                                        on="seqID",
                                        how="left"
                                )\
                            .with_columns(
                                        pl.max_horizontal(0, pl.col("start") - 1).alias("start_expanded"),
                                        pl.min_horizontal(pl.col("size"), pl.col("end") + 1).alias("end_expanded")
                                    )
    return merged_motifs_df

def fetch_extended_sequence(merged_motifs: pl.DataFrame, genome_fasta: str):
    """
    Fetches extended sequences for motif regions from a genome FASTA file.

    Args:
        merged_motifs (pl.DataFrame): DataFrame of motif regions with expanded coordinates.
        genome_fasta (str): Path to genome FASTA file.

    Returns:
        pd.DataFrame: DataFrame with extended motif sequences.
    """
    merged_motifs_bed = BedTool.from_dataframe(merged_motifs.select(["seqID", 
                                                                    "start_expanded", 
                                                                    "end_expanded", 
                                                                    "sequence"]).to_pandas()).sort()
    merged_motifs_seq_expanded = merged_motifs_bed.sequence(fi=genome_fasta,
                                                            tab=True,
                                                            name=True)
    new_lines = []
    with open(merged_motifs_seq_expanded.seqfn, 
              mode="r", 
              encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            data = line[0]
            seqID = data.split("::")[1].split(":")[0]
            coords = data.split(":")[-1]
            start, end = coords.split("-")
            start = int(start)
            end = int(end)
            sequence = line[1].lower()

            new_lines.append({
                        "seqID": seqID,
                        "start": start,
                        "end": end,
                        "sequence": sequence,
            })
    g4_df_exp = pd.DataFrame(new_lines)
    # g4_bed = BedTool.from_dataframe(g4_df_exp).sort()
    return g4_df_exp

def extend_mutations(mutations_df_expanded: pl.DataFrame, 
                     genome_fasta: str) -> pl.DataFrame:
    """
    Extends mutation regions and annotates trinucleotide context from the genome FASTA.

    Args:
        mutations_df_expanded (pl.DataFrame): DataFrame of expanded mutation regions.
        genome_fasta (str): Path to genome FASTA file.

    Returns:
        pl.DataFrame: DataFrame with trinucleotide context and annotations.
    """
    mutations_bed = BedTool.from_dataframe(mutations_df_expanded.select(["seqID",
                                                                         "start",
                                                                         "end",
                                                                         "reference",
                                                                         "variant",
                                                                         ]).to_pandas()
                                           ).sort()
    mutation_bed_seq = mutations_bed.sequence(
                                        fi=str(genome_fasta), 
                                        tab=True, 
                                        name=True)
    trinucleotides = [] 

    # fetch the extended mutation sequence
    with open(mutation_bed_seq.seqfn, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("\t")
            data = line[0]
            ref = data.split('::')[0]
            seqID = data.split('::')[1].split(':')[0]
            if seqID == "chrY":
                continue
            coords = data.split(':')[-1]
            x, y = coords.split('-')
            
            sequence = line[1].lower()
            assert sequence[1] == ref.lower()
            trinucleotides.append(
                            {
                             "seqID": seqID,
                             "start": x,
                             "end": y,
                             "ref": ref,
                             "sequence": sequence,
                            }
            )
            
    trinucleotides = pl.DataFrame(trinucleotides)
    def replace_with_complement(seq: str) -> str:
        seq = seq.lower()
        # rev_comp = str(Seq(seq).reverse_complement())
        rev_comp = reverse_complement(seq)
        if seq <= rev_comp:
            return seq
        return rev_comp

    # merge back to the original dataframe to have all the information (reference > variant)
    trinucleotides_annotated = trinucleotides.with_columns(
                                        pl.col("start").cast(pl.Int32),
                                        pl.col("end").cast(pl.Int32)
                                    )\
                                    .join(
                                        mutations_df_expanded.select(["seqID", 
                                                                      "start", 
                                                                      "end", 
                                                                      "reference", 
                                                                      "variant", 
                                                                      "AF"]),
                                        how="left",
                                        on=["seqID", "start", "end"]
                )\
                .with_columns(
                        pl.col("sequence").map_elements(replace_with_complement, 
                                                        return_dtype=str)
                        .alias("canonical_sequence")
                )\
                .with_columns(
                        (pl.col("sequence").str.slice(0, 1) + pl.col("variant").str.to_lowercase() + pl.col("sequence").str.slice(2, 1))
                                        .alias("mutant"),
                        pl.col("sequence").str.slice(1, 1).str.to_uppercase()
                                        .alias("ref_fetched")
                )
    if trinucleotides_annotated.filter(pl.col("ref_fetched") != pl.col("reference")).shape[0] > 0:
        raise ValueError("Something went wrong...")

    return trinucleotides_annotated

def reverse_complement(seq: str) -> str:
    """
    Returns the reverse complement of a DNA sequence (lowercase).

    Args:
        seq (str): DNA sequence (a, t, g, c).

    Returns:
        str: Reverse complement sequence.
    """
    return ''.join({'a': 't', 
                    'g': 'c', 
                    'c': 'g', 
                    't': 'a'}[x] for x in seq)[::-1]

def count_trinucleotides(motifs_df: pl.DataFrame) -> pl.DataFrame:
    """
    Counts the occurrences of each trinucleotide in the provided motif sequences.

    Args:
        motifs_df (pl.DataFrame): DataFrame with a 'sequence' column.

    Returns:
        pl.DataFrame: DataFrame with trinucleotide counts.
    """
    # count total trinucleotides
    if "sequence" not in motifs_df.columns:
        raise KeyError("Column `sequence` does not exist in the dataframe.")

    base_trinucleotide_counts = defaultdict(int)
    expanded_sequences = list(motifs_df["sequence"])

    for sequence in tqdm(expanded_sequences):
        sequence = sequence.lower()
        # if sequence.count("g") < sequence.count("c"):
            # sequence = str(Seq(sequence).reverse_complement())
            # sequence = reverse_complement(sequence)
        
        for i in range(len(sequence)-2):
            trinucleotide = sequence[i:i+3]
            rev_comp = reverse_complement(trinucleotide)
            if trinucleotide > rev_comp:
                trinucleotide = rev_comp
                
            base_trinucleotide_counts[trinucleotide] += 1
    base_trinucleotide_counts = pd.Series(base_trinucleotide_counts)\
                                    .to_frame('base_tri_counts')\
                                    .reset_index()\
                                    .rename(columns={"index": "canonical_sequence"})\
                                    .sort_values(
                                            by=['base_tri_counts'],
                                            ascending=False
                                    )
    return base_trinucleotide_counts

def identify_mutation_area(row: dict) -> str:
    """
    Identifies the mutation area (e.g., G-run or C-run) for a given row.

    Args:
        row (dict): Dictionary with mutation and sequence information.

    Returns:
        str: Mutation area label or empty string if not found.
    """
    row = dict(row)
    mut_start = row["mut_start"]
    start = row["start"]
    sequence = row["sequence"]
    if sequence.count("g") >= sequence.count("c"):
        motif = "g"
    else:
        motif = "c"
    gruns = re.finditer("%s{3,}" % motif, sequence)
    relative_mut_pos = mut_start + 1 - start
    for _, grun in enumerate(gruns, 0):
        start = grun.start()
        end = grun.end()
        if end == relative_mut_pos \
            or end == relative_mut_pos + 1 \
            or start == relative_mut_pos \
            or start == relative_mut_pos + 1:
            return "boundary"
        if relative_mut_pos < start - 1:
            return "loop"
        if start + 1 < relative_mut_pos < end - 1:
            return "grun"
    return "" # ValueError(f"Invalid area for sequence {sequence}, mutation {mut_start} and {start}.")

def find_gruns(sequence: str) -> int:
    """
    Finds the number of G-runs or C-runs in a sequence.

    Args:
        sequence (str): DNA sequence.

    Returns:
        int: Number of G-runs or C-runs.
    """
    if sequence.count("g") >= sequence.count("c"):
        motif = "g"
    else:
        motif = "c"
    return len(re.findall("%s{3,}" % motif, sequence))

def main():
    """
    Main workflow for trinucleotide enrichment analysis from motifs and mutations.

    Args:
        None

    Returns:
        None
    """
    import argparse
    parser = argparse.ArgumentParser(description="""Germ Hunter 
                                    is a command line utility that 
                                    calculates the Trinucleotide Enrichment given a bed file in respect to the 
                                    genome-wide mutation frequencies.""")

    parser.add_argument("vcf", type=str)
    parser.add_argument("motifs", type=str)
    parser.add_argument("fasta", type=str)
    parser.add_argument("gw_tri_counts", type=str)
    parser.add_argument("genome_size", type=str)
    parser.add_argument("--separator", type=str, default="\t")
    parser.add_argument("--outdir", type=str)
    parser.add_argument("--mutation", 
                        type=str, 
                        choices=["snp", "smalldel", "smallins", "del", "ins"], 
                        default="snp")

    args = parser.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)
    print(colored(f"Initialized Trinucleotide Model process for mutation `{args.mutation}`...", "green"))
    print(colored(f"Saving results to --> `{outdir}`.", "green"))

    ## LOAD MOTIFS
    motifs_df = pl.read_csv(args.motifs, 
                            columns=["seqID", "start", "end", "sequence"],
                            separator=args.separator
                            )
    motifs_bed = BedTool.from_dataframe(motifs_df.to_pandas())\
                        .sort()\
                        .merge(
                                c=["2", "4"],
                                o=["collapse", "collapse"],
                                delim=","
                            )
    merged_motifs_df = pd.read_csv(motifs_bed.fn,
                                sep="\t",
                                header=None,
                                names=["seqID", "start", "end", "multiple_starts", "multiple_sequences"],
                            )
    merged_motifs_df = pl.from_pandas(merged_motifs_df)
    merged_motifs_df = merge_motifs(merged_motifs_df, 
                                    args.genome_size)
    merged_motifs_df = fetch_extended_sequence(merged_motifs_df, genome_fasta=args.fasta)
    merged_motifs_bed = BedTool.from_dataframe(merged_motifs_df).sort()

    # -- This calculates all trinucleotides within Motif Region E --
    motif_tri_counts = count_trinucleotides(merged_motifs_df)
    motif_tri_counts = pl.from_pandas(motif_tri_counts)

    # -- Load base trinucleotides counts --
    # This is the same as the step above but for genome-wide area
    gw_tri_counts = pl.read_csv(
                                args.gw_tri_counts, 
                                separator=" ",
                                has_header=False,
                                new_columns=["canonical_sequence", "gw_counts"]
                        )\
                        .with_columns(pl.col("canonical_sequence").str.to_lowercase())

    # Find mutated trinucleotides within the expanded motifs
    # -- Load Mutations --
    mutation_df = pl.read_csv(
                     args.vcf, 
                     comment_prefix="#",
                     separator=",",
                     columns=["seqID", "start", "end", "mutation", "reference", "variant", "AF"]
        )
    mutation_df = mutation_df.filter(pl.col("mutation") == args.mutation)\
                             .with_columns(pl.lit("+").alias("strand"))\
                             .drop(["mutation"])
    # -- Expand Mutations by one base pair -- 
    e = Expander(window_size=1)
    mutation_df_expanded = e.expand_windows(mutation_df, loci="start")\
                            .with_columns(length=pl.col("end") - pl.col("start"))\
                            .filter(pl.col("length") == 3)
    
    # -- Extend Mutations on MOTIFS --
    # MOTIFS: We calculate the mutated trinucleotide sequences GENOME WIDE
    tri_gw_mut_ratio = extend_mutations(mutation_df_expanded, 
                                        genome_fasta=args.fasta)
    
    # First downstream analysis
    # -- Calculate base mutagenicity ratio for each canonical trinucleotide
    # -- Mutagenicity = # Trinucleotide Mutated  / # Trinucleotide Occurrences
    gw_trinucleotides_mutated = tri_gw_mut_ratio\
            .group_by("canonical_sequence")\
            .agg(
                    pl.col("reference").count().alias("totalTimesMutated")
                )\
            .join(
                gw_tri_counts,
                on="canonical_sequence",
                how="left"
            )\
            .with_columns(
                (pl.col("totalTimesMutated") / pl.col("gw_counts")).alias("mutagenicity")
            )\
            .sort(['mutagenicity'], descending=True)

    # -- Step 1: Fetch the motifs with mutations
    # -- Step 2: Fetch the mutated sequences
    # -- Step 3: Count the base counts from the mutated motifs
    # -- Step 4: Iterate over the mutated motifs and calculated
    
    # -- Step 1 -- 
    mut_expanded_bed = BedTool.from_dataframe(mutation_df_expanded.to_pandas()).sort()
    motifs_to_mut = pl.read_csv(
                            merged_motifs_bed.intersect(mut_expanded_bed, wo=True).fn,
                            has_header=False,
                            separator="\t",
                            new_columns=["seqID", "start", "end", "sequence"] \
                                    + ["chromosome", 
                                       "mut_start", 
                                       "mut_end", 
                                       "AF", 
                                       "ref", 
                                       "var", 
                                       "strand", 
                                       "length", 
                                       "overlap"]
                        )\
                        .filter(pl.col("overlap") == 3)
    
    # keep global mutated motifs
    # track transitions
    # X > Y , AXB > AYB
    # X > Y ---> corresponds to 
    
    def reverse_with_complement(seq: str) -> str:
        rev_comp = reverse_complement(seq)
        if rev_comp < seq:
            seq = rev_comp
        return seq

    # motif_trinucleotides_mutated = defaultdict(int)
    # mut_transitions = defaultdict(lambda : defaultdict(int))
    motifs_to_mut = motifs_to_mut.with_columns(
                                                transition=pl.col("ref") + ">" + pl.col("var"),
                                                total_gruns=pl.col("sequence").map_elements(find_gruns,
                                                                                            return_dtype=int)
                                  )\
                                 .with_columns(
                                         pl.struct(["sequence",
                                                    "start",
                                                    "end",
                                                    "mut_start",
                                                    "mut_end"
                                                    ])
                                        .map_elements(lambda row: row["sequence"][row["mut_start"] - row["start"]: row["mut_end"] - row["start"]],
                                                                    return_dtype=str)
                                        .alias("tri_sequence")
                                 )\
                                .with_columns(
                                        tri_length=pl.col("tri_sequence").map_elements(len, return_dtype=int)
                                 )\
                                .filter(pl.col("tri_length") == 3)\
                                .with_columns(
                                            canonical_sequence=pl.col("tri_sequence")
                                            .map_elements(reverse_with_complement,
                                                          return_dtype=str)
                                 )\
                                .drop(["sequence"])\

    # mutated_motif_tri_counts = pl.from_pandas(
    #           count_trinucleotides(motifs_to_mut.rename({"tri_sequence": "sequence"}))
    #        )

    # -- Calculate Fold Enrichment --
    # Downstream I: Fold Enrichment for each mutated trinucleotide sequence
    # IN / Genome - Wide ratio
    # motifs_to_mut.group_by(["mutation_area"])\
    #             .agg(
    #                      pl.col("ref").count().alias("times_mutated"),
    #                      pl.col("total_gruns").sum().alias("total_gruns")
    #              )\
    #             .write_csv(f"{outdir}/mutation_area_counts.csv")
    
    motif_trinucleotides_mutated = motifs_to_mut.group_by(["canonical_sequence"])\
                                        .agg(
                                                pl.col("ref").count().alias("tri_mut_count"),
                                        )\
                                        .join(
                                                motif_tri_counts,
                                                how="left",
                                                on="canonical_sequence",
                                        )\
                                        .with_columns(
                                                mutagenicity=pl.col("tri_mut_count") / pl.col("base_tri_counts")
                                            )\
                                        .join(
                                                gw_trinucleotides_mutated,
                                                on="canonical_sequence",
                                                how="full",
                                                suffix="_gw"
                                        )\
                                        .with_columns(
                                                fold_enrichment=pl.col("mutagenicity") / pl.col("mutagenicity_gw")
                                        )\
                                        .drop(["canonical_sequence_gw"])\
                                        .filter(pl.col("canonical_sequence").is_not_null())\
                                        .with_columns(
                                                pl.struct([
                                                    "tri_mut_count",
                                                    "base_tri_counts",
                                                    "totalTimesMutated",
                                                    "gw_counts",
                                                    "mutagenicity"])
                                                .map_elements(contigency, return_dtype=float)
                                                .alias("p_value")
                                        )
    motif_trinucleotides_mutated = motif_trinucleotides_mutated\
                                        .with_columns(
                                                    (pl.col("p_value") * motif_trinucleotides_mutated.shape[0]).alias("adj_pvalue")
                                        )\
                                        .with_columns(
                                                pl.col("adj_pvalue").map_elements(map_stars, return_dtype=str)
                                                    .alias("significance")
                                        )\
                                        .with_columns(
                                                canonical_sequence=pl.col("canonical_sequence").str.to_uppercase()
                                        )\
                                        .sort(["fold_enrichment"], descending=False)

    ## -- Save Enrichment Results --
    motif_trinucleotides_mutated.write_csv(f"{outdir}/trinucleotide_model_motif_enrichment_matrix.{args.mutation}.csv")

    ## Downstream analysis II
    canonical_transitions = [
                             "A>G",
                             "A>C",
                             "A>T",
                             "C>T",
                             "C>G",
                             "C>A"
                             ]
    transition_mapping = {
                          "T>C": "A>G",
                          "T>G": "A>C",
                          "T>A": "A>T",
                          "G>A": "C>T",
                          "G>C": "C>G",
                          "G>T": "C>A"
                          }
    gw_transition_df = tri_gw_mut_ratio\
                            .with_columns(
                                    transition=pl.col("reference") + ">" + pl.col("variant")
                            )\
                            .with_columns(
                                    canonical_transition=pl.col("transition")
                                    .map_elements(lambda transition: transition_mapping.get(transition, transition),
                                                  return_dtype=str)
                             )\
                            .group_by(["canonical_transition", 
                                       "canonical_sequence"], maintain_order=True)\
                            .agg(
                                    pl.col("reference").count().alias("transition_occurrences")
                            )

    motifs_transition_df = motifs_to_mut\
                            .with_columns(
                                    canonical_transition=pl.col("transition")
                                    .map_elements(lambda transition: transition_mapping.get(transition, transition),
                                                  return_dtype=str)
                             )\
                            .group_by(["canonical_transition", 
                                       "canonical_sequence"], maintain_order=True)\
                            .agg(
                                    pl.col("ref").count().alias("transition_occurrences")
                            )\
                            .join(
                                    gw_transition_df,
                                    on=["canonical_transition", "canonical_sequence"],
                                    how="full",
                                    suffix="_gw"
                            )\
                            .with_columns(
                                    transition_enrichment=pl.col("transition_occurrences") / pl.col("transition_occurrences_gw")
                            )
    motifs_transition_df.write_csv(f"{outdir}/trinucleotide_model_transition_enrichment.{args.mutation}.csv")
    # << END
    print(colored(f"Trinucleotide model pipeline has completed successfully.", "green"))
    breakpoint()
                                           

if __name__ == "__main__": 

    main()

import math
import re
from termcolor import colored
import itertools
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import seaborn as sns
from collections import defaultdict

class Haplotypes:

    def __init__(self, indir: str,
                        hprc_metadata: str,
                        valid_haplotypes: str,
                        mode: str,
                        reference: str,
                        outdir: str,
                        test: int = 0,
                        test_samples: int = 5):
        self.indir = Path(indir).resolve()
        self.test = test
        self.test_samples = test_samples
        self.mode = mode
        if self.test:
            print(colored(f"WARNING! This is a test round! Only {self.test_samples} haplotypes will be processed.", "red"))

        if not self.indir.is_dir():
            raise ValueError(f"Directory {self.indir} does not exist.")

        self.valid_haplotypes = Path(valid_haplotypes).resolve()
        self.outdir = Path(outdir).resolve()
        self.outdir.mkdir(exist_ok=True)
        self.outdir = self.outdir.joinpath(self.mode)
        self.outdir.mkdir(exist_ok=True)
        print(f"Outsourcing results to ---> `{self.outdir}`.")
        print(f"Using mode --> `{self.mode}`.")
        
        # load haplotypes
        self.haplotypes = [haplotype.strip() for haplotype in self.valid_haplotypes.open().readlines()]
        self.total_valid_haplotypes = len(self.haplotypes)
        print(colored(f"Loaded {self.total_valid_haplotypes} total valid haplotypes.", "green"))
        self.haplotype_paths = self.load_haplotypes()
        self.total_haplotypes = len(self.haplotype_paths)

        if self.total_haplotypes == 0:
            raise ValueError("No haplotypes were detected.")
        print(colored(f"Loaded {self.total_haplotypes} haplotypes.", "green"))

        # load metadata
        self.hprc_metadata = hprc_metadata
        self.metadata = pd.read_table(hprc_metadata)\
                                .set_index("Sample")['Superpopulation'].dropna().to_dict()
        self.superpopulations = ["AFR", "AMR", "EAS", "SAS", "ASJ"]
        self.metadata = self.metadata | {"HG002":"ASJ", "HG005": "EAS", "NA21309": "AFR"}

        # load reference genome motifs
        self.reference = Path(reference).resolve()
        if not self.reference.is_file():
            raise FileNotFoundError(f"Could not locate reference genome at `{self.reference}`.")
        self.reference_df = self.load_reference()
        self.total_motifs = self.reference_df.shape[0]
        print(colored(f"Total reference motifs loaded: {self.total_motifs}.", "green"))

    def load_haplotypes(self) -> list[str]:
        """
        Loads haplotype file paths for valid haplotypes, skipping curated and duplicate entries.
        
        Args:
            None
        
        Returns:
            list[str]: List of file paths to valid haplotype CSV files.
        """
        haplotypes = {Path(str(file)\
                                .replace(".f1_v2.1_genomic-regex", "")\
                                .replace(".f1_v2_genomic-regex", "")\
                                .replace(".f1_v2", "")
                                ): file for file in self.indir.glob("*.csv") if "CHM" not in file.name}
        haplotype_paths = []
        sample_ids = defaultdict(set)
        for file_id, file in haplotypes.items():
            if file_id.name.startswith("HG002") or "cur" in file.name:
                print(colored(f"Skipping haplotype `{file}`. Probably curated (cur)?...", "red"))
                continue
            try:
                sample_id_code = file_id.name.split('.')[-4].split("_")[1]
            except IndexError as e:
                print(f"Failed processing file `{file}` with id `{file_id}`.")
                raise IndexError(e)
            haplotype_type = Haplotypes.find_type(file)
            if sample_id_code in sample_ids and haplotype_type in sample_ids[sample_id_code]:
                raise ValueError(f"Duplicate haplotype code id detected: '{sample_id_code}'.")
            sample_ids[sample_id_code].add(haplotype_type)
            if sample_id_code in self.haplotypes:
                haplotype_paths.append(file)
        return haplotype_paths

    def load_reference(self) -> pd.DataFrame:
        """
        Loads the reference genome motifs as a DataFrame and adds a canonical sequence column.
        
        Args:
            None
        
        Returns:
            pd.DataFrame: DataFrame with reference motifs and canonical sequences.
        """
        reference_df = pd.read_table(self.reference)
        reference_df.loc[:, "sequence"] = reference_df["sequence"].str.upper()
        reference_df.loc[:, "canonical_sequence"] = reference_df["sequence"].apply(Haplotypes.complement)
        return reference_df

    @staticmethod
    def reverse_complement(seq: str) -> str:
        """
        Returns the reverse complement of a DNA sequence.
        
        Args:
            seq (str): DNA sequence (A, T, G, C).
        
        Returns:
            str: Reverse complement sequence.
        """
        return ''.join({
                    'A': 'T', 
                    'T': 'A', 
                    'G': 'C', 
                    'C': 'G'}[x] for x in seq)[::-1]
    
    @staticmethod
    def complement(seq: str) -> str:
        """
        Returns the canonical sequence: the input if G count >= C count, else its reverse complement.
        
        Args:
            seq (str): DNA sequence (A, T, G, C).
        
        Returns:
            str: Canonical sequence.
        """
        total_g = seq.count('G')
        total_c = seq.count('C')
        if total_g >= total_c:
            return seq
        return Haplotypes.reverse_complement(seq)
    
    @staticmethod
    def find_type(sample: str) -> str:
        """
        Determines the haplotype type (maternal or paternal) from the sample name.
        
        Args:
            sample (str): Sample name or file name.
        
        Returns:
            str: 'maternal' or 'paternal'.
        """
        if not isinstance(sample, str):
            sample = str(sample)

        if "mat" in sample:
            return "maternal"
        if "pat" in sample:
            return "paternal"
        raise ValueError()

    def load_data(self) -> dict[str, set[str]]:
        """
        Loads and processes haplotype motif sequences for all valid samples.
        
        Args:
            None
        
        Returns:
            dict[str, set[str]]: Dictionary mapping sample IDs to sets of motif sequences.
        """
        # this needs to be run on HPC - huge amount of data
        sequences_per_sample = {}
        total_g4 = {}

        for idx, sample in tqdm(enumerate(self.haplotype_paths, 0), total=len(self.haplotype_paths)):
            temp = pd.read_csv(sample, usecols=['sequence'])
            temp = temp[~temp['sequence'].str.contains('N')]
            temp['sequence'] = temp['sequence'].str.upper()
            temp['sequence'] = temp['sequence'].apply(Haplotypes.complement)

            total = temp.shape[0]
            if "CHM13" in sample.name or "chm13" in sample.name:
                sample_id = "chm13v2"
            else:
                sample_id = sample.name.split('.')[1].split('_')[1] + '.' + Haplotypes.find_type(sample.name)
                
            # total_g4[sample_id] = len(temp['sequence'])
            if "CHM13" in sample.name or "cur" in sample.name or "chm13" in sample.name:
                print(f"Skipping haplotype {sample}.")
                continue

            # sample_id = sample.name.split('.')[0] + '.' + find_type(sample.name)
            assert sample_id.split('.')[0] in self.metadata
            sequences_per_sample.update({sample_id: set(temp['sequence'])})
            total_g4.update({sample_id: total})

            if self.test and idx > self.test_samples:
                break

        dest = f"{self.outdir}/total_g4_per_haplotype_{self.mode}.csv"
        with open(dest, mode="w", encoding="utf-8") as f:
            f.write("haplotype,total_g4\n")
            for sample_id, total in total_g4.items():
                f.write(f"{sample_id},{total}\n")
        print(colored(f"Saved total g4 for mode {self.mode}!", "green"))
        print(colored(f"Total haplotypes loaded: {len(sequences_per_sample)}.", "green"))
        return sequences_per_sample

    def calculate_total_motifs_across_haplotypes(self, sequences_per_sample: dict[str, set[str]], saveas: bool = True) -> pd.DataFrame:
        """
        For each haplotype, extracts the total number of unique motifs.
        
        Args:
            sequences_per_sample (dict[str, set[str]]): Dictionary mapping sample IDs to sets of motif sequences.
            saveas (bool, optional): Whether to save the result as CSV. Defaults to True.
        
        Returns:
            pd.DataFrame: DataFrame with total unique motifs per haplotype and population info.
        """
        total_df = {}
        for key, seq in sequences_per_sample.items():
            total_df[key] = len(seq)
        total_df = (
                        pd.Series(total_df)
                            .sort_values(ascending=False)
                            .to_frame(name="total_unique_motifs")
                )
        total_df['pop'] = total_df.index.map(lambda x: self.metadata.get(x.split('.')[0], 'chm13v2'))
        # total['total'] = total.index.map(total_g4)
        # total['uniqueness'] = total['total_g4'].div(total['total'])
        if saveas:
            dest = f"{self.outdir}/total_motifs_across_haplotypes_{self.mode}.csv"
            total_df.to_csv(dest,
                                                  sep=",",
                                                  index=False,
                                                  mode="w",
                                                  header=True
                                                  )
            print(colored(f"Total motifs across haplotypes data have been saved at `{dest}`.", "green"))
        return total_df

    def load_motifs_across_haplotypes(self, sequences_per_sample: dict[str, set[str]]) -> dict[str, set[str]]:
        """
        Loads motifs across haplotypes, grouped by assembly/sample.
        
        Args:
            sequences_per_sample (dict[str, set[str]]): Dictionary mapping sample IDs to sets of motif sequences.
        
        Returns:
            dict[str, set[str]]: Dictionary mapping assembly/sample to set of motifs.
        """
        motifs_per_assembly = defaultdict(set)
        for haplotype in tqdm(sequences_per_sample):
            seq = sequences_per_sample[haplotype]
            sample = haplotype.split('.')[0]
            motifs_per_assembly[sample] = motifs_per_assembly[sample].union(seq)
        print(f"Loaded {len(motifs_per_assembly)} motifs per assembly.")
        return motifs_per_assembly

    def shared_with_reference_genome(self, sequences_per_sample: dict, saveas: bool = True) -> pd.DataFrame:
        """
        For each haplotype, extracts the percentage of motifs from the reference genome found in the haplotype.
        Removes non-autosomal chromosomes for this analysis.
        
        Args:
            sequences_per_sample (dict): Dictionary mapping sample IDs to sets of motif sequences.
            saveas (bool, optional): Whether to save the result as CSV. Defaults to True.
        
        Returns:
            pd.DataFrame: DataFrame with sharing statistics per haplotype and chromosome.
        """
        # fetch autosomal motifs
        autosomal_motifs = self.reference_df.query("seqID != 'chrX' & seqID != 'chrY'").dropna(subset=['seqID'])
        # fetch autosomal chromosomes
        sequence_ids = set(autosomal_motifs["seqID"].unique())
        total_sequence_ids = len(sequence_ids)
        print(f"Total autosomal motifs: {autosomal_motifs.shape[0]}.")
        print(f"Total autosomal chromosomes: {total_sequence_ids}.")

        chm13v2_sequences_specific_seqID = dict()
        total_ref = dict()
        for seqID in tqdm(sequence_ids, total=total_sequence_ids):
            chm13v2_sequences_specific_seqID[seqID] = set(autosomal_motifs[autosomal_motifs["seqID"] == seqID]["canonical_sequence"])
            total_ref[seqID] = len(chm13v2_sequences_specific_seqID[seqID])

        common_with_reference_specific = defaultdict(list)
        for sample in tqdm(sequences_per_sample):
            seq = sequences_per_sample[sample]
            for seqID in sequence_ids:
                chm13v2_sequences_specific = chm13v2_sequences_specific_seqID[seqID]

                mutual = seq.intersection(chm13v2_sequences_specific)
                shared = seq.union(chm13v2_sequences_specific)
                jaccard_index = len(mutual) / len(shared)

                common_with_reference_specific["haplotype"].append(sample)
                common_with_reference_specific["jaccard"].append(jaccard_index)
                common_with_reference_specific["union"].append(len(shared))
                common_with_reference_specific["total_ref"].append(total_ref[seqID])
                common_with_reference_specific["pop"].append(self.metadata[sample.split('.')[0]])
                common_with_reference_specific["mutual"].append(len(mutual))
                common_with_reference_specific["total"].append(len(seq))
                common_with_reference_specific["seqID"].append(seqID)

        common_with_reference_specific = pd.DataFrame(common_with_reference_specific)
        unique_g4_motifs_per_seqID = autosomal_motifs.groupby("seqID").agg({"canonical_sequence": "nunique"})["canonical_sequence"].to_dict()
        common_with_reference_specific.loc[:, "unique_ref_motifs"] = common_with_reference_specific["seqID"].map(unique_g4_motifs_per_seqID)
        common_with_reference_specific.loc[:, "shared_with_ref_perc"] = 1e2 * common_with_reference_specific["mutual"] / common_with_reference_specific["total_ref"]
        if saveas:
            dest = f"{self.outdir}/common_with_reference_specific_motifs_{self.mode}.csv"
            common_with_reference_specific.to_csv(dest,
                                                  sep=",",
                                                  index=False,
                                                  mode="w",
                                                  header=True
                                                  )
            print(colored(f"Motifs shared with reference genome data have been saved at `{dest}`.", "green"))
        return common_with_reference_specific 

    def pairwise_shared_motifs(self, sequences_per_sample: dict[str, set[str]], saveas: bool = True) -> pd.DataFrame:
        """
        Extracts Jaccard index for unique motifs by conducting pairwise comparisons between haplotypes.
        
        Args:
            sequences_per_sample (dict[str, set[str]]): Dictionary mapping sample IDs to sets of motif sequences.
            saveas (bool, optional): Whether to save the result as CSV. Defaults to True.
        
        Returns:
            pd.DataFrame: DataFrame with pairwise Jaccard indices between haplotypes.
        """
        jaccard_indices = {}
        samples = sequences_per_sample.keys()

        for sample1, sample2 in tqdm(itertools.product(samples, samples), total=len(samples) ** 2):
            jaccard = len(sequences_per_sample[sample1].intersection(sequences_per_sample[sample2])) / len(sequences_per_sample[sample1].union(sequences_per_sample[sample2]))
            jaccard_indices[sample1, sample2] = jaccard
        jaccard_df = (
                    pd.Series(jaccard_indices)
                    .to_frame(name="jaccard")
                    .reset_index()
                    .rename(columns={"level_0": "sampleA", 
                                     "level_1": "sampleB"
                        })
                    )
        if saveas:
            dest = f"{self.outdir}/jaccard_pairwise_haplotypes_unique_motifs_{self.mode}.csv"
            jaccard_df.to_csv(dest,
                            sep=",",
                            mode="w",
                            index=False,
                            header=True
                        )
            print(colored(f"Pairwise shared motifs data have been saved at `{dest}`.", "green"))
        return jaccard_df

    def extract_unique_sequences(self, sequences_per_sample: dict[str, set[str]], 
                                       sex_distinction: bool = False, 
                                       saveas: bool = True) -> pd.DataFrame:
        """
        Extracts unique sequences in each haplotype by removing all sequence motifs encountered in the remaining haplotypes.
        
        Args:
            sequences_per_sample (dict[str, set[str]]): Dictionary mapping sample IDs to sets of motif sequences.
            sex_distinction (bool, optional): Whether to distinguish by sex. Defaults to False.
            saveas (bool, optional): Whether to save the result as CSV. Defaults to True.
        
        Returns:
            pd.DataFrame: DataFrame with unique sequence counts per haplotype.
        """
        unique_sequences = defaultdict(int)
        for sample_A in tqdm(sequences_per_sample, leave=True):
            if "chm" in sample_A:
                continue
            sample_sequences_A = sequences_per_sample[sample_A]
            for sample_B in sequences_per_sample:
                if "chm" in sample_B:
                    continue
                if sample_A == sample_B:
                    continue
                    
                sample_sequences_A = sample_sequences_A - sequences_per_sample[sample_B]
            remaining_sequences = len(sample_sequences_A)
            unique_sequences[sample_A] = remaining_sequences
        unique_sequences_df = (
                                pd.Series(unique_sequences)
                                    .to_frame(name="unique_seq")
                                    .reset_index()
                                    .rename(columns={"index": "Haplotype"})\
                                    .sort_values(by=["unique_seq"], ascending=False)
                                    .reset_index(drop=True)
                        )
        if saveas:
            if sex_distinction:
                dest = f"{self.outdir}/unique_sequences_per_haplotype_{self.mode}.csv"
            else:
                dest = f"{self.outdir}/unique_sequences_per_haplotype_{self.mode}.no-sex-distinction.csv"

            unique_sequences_df.to_csv(dest,
                                       sep=",",
                                       mode="w",
                                       header=True,
                                       index=False
                                       )
            print(colored(f"Unique sequence motifs across haplotypes data have been saved at `{dest}`.", "green"))
        return unique_sequences_df


    @staticmethod
    def extract_loop_ratio(sequence: str) -> float:
        """
        Calculates the loop ratio (percentage of non-G/C bases) in a motif sequence.
        
        Args:
            sequence (str): Motif sequence.
        
        Returns:
            float: Loop ratio as a percentage.
        """
        sequence = sequence.upper()
        leader = 'G' if sequence.count('G') >= sequence.count('C') else 'C'
        total_length = len(sequence)
        seq = re.sub("%s{2,}" % leader, "", sequence)
        loop_size = len(seq)
        return 1e2 * loop_size / total_length

    def shared_across_haplotypes(self, sequences_per_sample: dict[str, set[str]], saveas: bool = True) -> pd.DataFrame:
        """
        For each motif, extracts the proportion of haplotypes in which it was encountered.
        
        Args:
            sequences_per_sample (dict[str, set[str]]): Dictionary mapping sample IDs to sets of motif sequences.
            saveas (bool, optional): Whether to save the result as CSV. Defaults to True.
        
        Returns:
            pd.DataFrame: DataFrame with sharing proportion, sequence length, and loop ratio for each motif.
        """
        g4_counter = defaultdict(int)
        for sample in tqdm(sequences_per_sample, total=len(sequences_per_sample)):
            if "chm13" in sample or "CHM13" in sample:
                continue
            for g4 in sequences_per_sample[sample]:
                g4_counter[Haplotypes.complement(g4)] += 1
        g4_counter = (
                    pd.Series(dict(g4_counter))
                    .to_frame(name='shared_proportion')
                    .reset_index()
                    .rename(columns={"index": "sequence"})
                    )
        g4_counter.loc[:, "length"] = g4_counter["sequence"].apply(len)
        g4_counter.loc[:, "loop_ratio"] = g4_counter["sequence"].apply(Haplotypes.extract_loop_ratio)

        if saveas:
            dest = f"{self.outdir}/sequence_shared_across_X_haplotypes_and_loop_size_{self.mode}.csv"
            g4_counter.to_csv(dest,
                              mode="w",
                              sep=",",
                              index=False,
                              header=True
                            )
            print(colored(f"Unique sequence motifs shared with X% haplotypes data have been saved at `{dest}`.", "green"))
            
            # save haplotype counts per shared proportion
            dest = f"{self.outdir}/total_counts_of_sequences_shared_across_X_haplotypes_{self.mode}.csv"
            g4_counter.groupby("shared_proportion")\
                                .agg(sample_counts=("sequence", "count"))\
                                .assign(log10_sample_counts=lambda ds: ds['sample_counts'].apply(lambda c: math.log(c, 10)))\
                                .to_csv(dest,
                                        mode="w",
                                        sep=",",
                                        index=True,
                                        header=True
                                )
        return g4_counter

    def extract_all(self):
        """
        Extracts all statistic tables for the haplotypes by running all extraction methods in sequence.
        
        Args:
            None
        
        Returns:
            None
        """
        sequences_per_sample = self.load_data()
        motifs_per_assembly = self.load_motifs_across_haplotypes(sequences_per_sample=sequences_per_sample)
        print(colored("Process has been initialized.", "green"))

        # calculate total number of unique motifs across each haplotype
        self.calculate_total_motifs_across_haplotypes(sequences_per_sample=sequences_per_sample, saveas=True)
        
        # extract the jaccard index of motifs for each haplotype that are being shared with the reference genome
        # but focusing exclusively to autosomal chromosomes 
        self.shared_with_reference_genome(sequences_per_sample=sequences_per_sample, saveas=True)

        # extract unique sequencse across haplotypes but no distinction based on sex
        self.extract_unique_sequences(sequences_per_sample=motifs_per_assembly, sex_distinction=False, saveas=True)

        # extract unique sequences across haplotypes but distinction based on sex
        self.extract_unique_sequences(sequences_per_sample=sequences_per_sample, sex_distinction=True, saveas=True)

        # extract the X% of haplotypes that contain a certain motif
        self.shared_across_haplotypes(sequences_per_sample=sequences_per_sample, saveas=True)

        # extract the jaccard index for haplotype-to-haplotype pairwise comparisons
        self.pairwise_shared_motifs(sequences_per_sample=sequences_per_sample, saveas=True)

        print(colored("Process has been completed succesfully.", "green"))
        

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description="""Haplotype motif extraction script.""")
    parser.add_argument("--outdir", type=str, default="pangenomes")
    parser.add_argument("--mode", type=str, default="G4HUNTER", choices=["REGEX", "G4HUNTER"])
    parser.add_argument("--valid_haplotypes", type=str, default="/storage/home/nmc6088/restored/valid_haplotyles.txt")
    parser.add_argument("--hprc_metadata", type=str, default="/storage/group/izg5139/default/nicole/datasets/hprc_year1_sample_metadata.txt")
    parser.add_argument("--test", type=int, default=0, choices=[0, 1])

    args = parser.parse_args()

    if args.mode == "G4HUNTER":
        reference = "/storage/group/izg5139/default/nicole/MirrorRTR/g4_results/chm13v2_g4hunter.txt"
        indir = "/storage/group/izg5139/default/nicole/pangenome_extractions/g4_out_v2/haplotypes"
    else:
        reference = "/storage/group/izg5139/default/nicole/MirrorRTR/chm13v2_regex_motifs.txt"
        indir = "/storage/group/izg5139/default/nicole/pangenome_extractions/regex_out_v2/haplotypes"

    haplotypes = Haplotypes(mode=args.mode, 
                            outdir=args.outdir,
                            indir=indir,
                            test=args.test,
                            valid_haplotypes=args.valid_haplotypes,
                            reference=reference,
                            hprc_metadata=args.hprc_metadata
                            )
    haplotypes.extract_all()

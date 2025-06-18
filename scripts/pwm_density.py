import numpy as np
import pandas as pd
import polars as pl
from typing import Callable, Optional, Iterator
from abc import abstractmethod 
from tqdm import tqdm
import matplotlib.pyplot as plt
from seaborn import color_palette

class PWMExtractor:
    """
    Extracts position weight matrix (PWM) densities and related statistics from motif data.
    """

    def __init__(self) -> None:
        """
        Initializes the PWMExtractor with nucleotide set.

        Args:
            None
        Returns:
            None
        """
        self.nucleotides = "agct"

    @staticmethod
    def invert(nucleotide: str) -> str:
        """
        Returns the complement of a nucleotide (lowercase).

        Args:
            nucleotide (str): Input nucleotide ('a', 't', 'g', 'c').

        Returns:
            str: Complement nucleotide.
        """
        match nucleotide:
            case 'a':
                return 't'
            case 't':
                return 'a'
            case 'g':
                return 'c'
            case 'c':
                return 'g'
            case _:
                raise ValueError(f'Unknown nucleotide {nucleotide}.')

    def extract_template_density(self, intersect_df: pl.DataFrame, 
                                        window_size: int, 
                                        enrichment: bool = False,
                                        return_frame: bool = True,
                                 ) -> pl.DataFrame:
        """
        Extracts template and non-template density profiles from intersected motif data.

        Args:
            intersect_df (pl.DataFrame): DataFrame of intersected motif data.
            window_size (int): Window size for density extraction.
            enrichment (bool, optional): Whether to normalize as enrichment. Defaults to False.
            return_frame (bool, optional): Whether to return as DataFrame. Defaults to True.

        Returns:
            pl.DataFrame: DataFrame with density profiles for template and non-template strands.
        """
        total_counts = {
                        "Template": np.zeros(2*window_size+1),
                        "Non-Template": np.zeros(2*window_size+1)
                    }
        total_overlap = 0
        for row in intersect_df.iter_rows(named=True):
            compartment_strand = row['strand']
            if compartment_strand == "?":
                continue
            start = int(row["start"])
            end = int(row["end"])
            motif_start = int(row["motif_start"])
            motif_end = int(row["motif_end"])
            if "strand_polarity" in row:
                polarity = row["strand_polarity"]
            else:
                motif_strand = row["motif_strand"]
                polarity = "Non-Template" if  motif_strand == compartment_strand else "Template"
            temp_counts = np.zeros(2*window_size+1)

            overlap = int(row['overlap'])
            total_overlap += overlap
            # [O-W, O+W+1)
            # len([O-W, O+W+1)) = 2W+1
            # O-W = start = origin - window
            # MS' = MS - S (MS = motif start)
            # ME' = ME - S (ME = motif end)
            origin = end - window_size - 1
            L = max(0, window_size - (origin - motif_start))
            U = min(2 * window_size + 1, window_size - (origin - motif_end))
            assert L <= U
            overlap_start = max(motif_start, start)
            overlap_end = min(motif_end, end)
            overlap_length = overlap_end - overlap_start
            assert overlap == overlap_length == U - L

            temp_counts[L:U] += 1
            if compartment_strand == "-":
                temp_counts = temp_counts[::-1]
            total_counts[polarity] += temp_counts

        # honor contract
        assert total_overlap == np.sum(total_counts["Template"]) + np.sum(total_counts["Non-Template"]), f"Overlap: {total_overlap} vs. Calculated overlap {np.sum(total_counts)}."
        if enrichment:
            name = "Enrichment"
            for typ in total_counts:
                total_counts[typ] = total_counts[typ] / np.mean(total_counts[typ])
        else:
            name = "Occurrences"
        if not return_frame:
            return total_counts

        for polarity in total_counts:
            total_counts[polarity] = pl.Series(total_counts[polarity])\
                                .to_frame(name=name + f"_{polarity}")
            if name == "Occurrences":
                total_counts[polarity] = total_counts[polarity].cast(pl.Int32)
        polarity_df = pl.concat([
                                 total_counts["Template"],
                                 total_counts["Non-Template"]
                                ], how="horizontal")
        return polarity_df

    @staticmethod
    def bayes_estimator(counts: int, total_counts: int, total_bins: int) -> float:
        """
        Computes the Bayesian estimator for motif counts.

        Args:
            counts (int): Motif count.
            total_counts (int): Total motif counts.
            total_bins (int): Number of bins.

        Returns:
            float: Bayesian estimate.
        """
        return (counts+1)/(total_counts+total_bins)

    @staticmethod
    def expected_value(counts: int, total_counts: int, total_bins: int) -> float:
        """
        Computes the expected value using the Bayesian estimator.

        Args:
            counts (int): Motif count.
            total_counts (int): Total motif counts.
            total_bins (int): Number of bins.

        Returns:
            float: Expected value.
        """
        return PWMExtractor.bayes_estimator(counts, total_counts, total_bins)

    @staticmethod
    def enrichment(counts: int, total_counts: int, total_bins: int) -> float:
        """
        Computes the enrichment value for motif counts.

        Args:
            counts (int): Motif count.
            total_counts (int): Total motif counts.
            total_bins (int): Number of bins.

        Returns:
            float: Enrichment value.
        """
        return counts/(total_counts * PWMExtractor.bayes_estimator(counts, total_counts, total_bins))

    def get_relative_positions(self, 
                               intersect_df: pl.DataFrame, 
                               window_size: int) -> Iterator[np.ndarray]:
        """
        Yields relative position arrays for each motif in the intersected data.

        Args:
            intersect_df (pl.DataFrame): DataFrame of intersected motif data.
            window_size (int): Window size for extraction.

        Returns:
            Iterator[np.ndarray]: Iterator of relative position arrays.
        """
        total_overlap = 0
        for row in intersect_df.iter_rows(named=True):
            compartment_strand = row['strand']
            if compartment_strand == "?":
                continue
            start = int(row['start'])
            end = int(row['end'])
            motif_start = int(row['motif_start'])
            motif_end = int(row['motif_end'])

            temp_counts = np.zeros(2*window_size+1)
            overlap = int(row['overlap'])
            total_overlap += overlap
            origin = end - window_size - 1
            L = max(0, window_size - (origin - motif_start))
            U = min(2 * window_size + 1, window_size - (origin - motif_end))

            assert L <= U
            overlap_start = max(motif_start, start)
            overlap_end = min(motif_end, end)
            overlap_length = overlap_end - overlap_start
            assert overlap == overlap_length
            assert overlap == U-L
            temp_counts[L:U] += 1

            if compartment_strand == "-":
                temp_counts = temp_counts[::-1]
            yield temp_counts

    def bootstrap(self, relative_positions: pd.DataFrame, 
                  N: int = 1_000,
                  lower_quantile: float = 0.025,
                  upper_quantile: float = 0.975) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Performs bootstrap resampling to estimate mean and confidence intervals.

        Args:
            relative_positions (pd.DataFrame): DataFrame of relative position densities.
            N (int, optional): Number of bootstrap samples. Defaults to 1000.
            lower_quantile (float, optional): Lower quantile for CI. Defaults to 0.025.
            upper_quantile (float, optional): Upper quantile for CI. Defaults to 0.975.

        Returns:
            tuple[pd.Series, pd.Series, pd.Series]: (mean, lower bound, upper bound) of densities.
        """
        bootstrap_df = []
        total_samples = relative_positions.shape[0]
        for _ in tqdm(range(N), leave=True, position=0):
            density = relative_positions.sample(total_samples, replace=True)\
                                        .sum(axis=0)
            density /= np.mean(density)
            bootstrap_df.append(density)
        bootstrap_df = pd.DataFrame(bootstrap_df)
        average = bootstrap_df.mean()
        lower_bound = bootstrap_df.quantile(lower_quantile)
        upper_bound = bootstrap_df.quantile(upper_quantile)
        return average, lower_bound, upper_bound
            
    def extract_density(self, intersect_df: pl.DataFrame, 
                        window_size: int,
                        return_array: bool = True,
                        return_frame: bool = False,
                        enrichment: bool = False,
                        ) -> list[int] | np.ndarray:
        """
        Extracts density profile for motifs in the intersected data.

        Args:
            intersect_df (pl.DataFrame): DataFrame of intersected motif data.
            window_size (int): Window size for density extraction.
            return_array (bool, optional): Whether to return as array. Defaults to True.
            return_frame (bool, optional): Whether to return as DataFrame. Defaults to False.
            enrichment (bool, optional): Whether to normalize as enrichment. Defaults to False.

        Returns:
            list[int] | np.ndarray: Density profile as list or array.
        """
        total_counts = np.zeros(2*window_size+1)
        total_overlap = 0
        for row in intersect_df.iter_rows(named=True):
            compartment_strand = row['strand']
            if compartment_strand == "?":
                continue
            start = int(row['start'])
            end = int(row['end'])
            # 10 -- 21 -- 31 ] 32 = 2 * w + 10 + 2
            # if end != window_size * 2 + 2 + start:
            #    print('woops?')
            #    end = start + window_size * 2 + 2
            
            motif_start = int(row['motif_start'])
            motif_end = int(row['motif_end'])
            temp_counts = np.zeros(2*window_size+1)
            overlap = int(row['overlap'])
            total_overlap += overlap
            origin = end - window_size - 1
            L = max(0, window_size - (origin - motif_start))
            U = min(2 * window_size + 1, window_size - (origin - motif_end))

            assert L <= U
            overlap_start = max(motif_start, start)
            overlap_end = min(motif_end, end)
            overlap_length = overlap_end - overlap_start
            assert overlap == overlap_length, f"{row}-{U}-{L}-{origin}"
            assert overlap == U-L, f"{row}-{U}-{L}-{origin}"
            temp_counts[L:U] += 1

            if compartment_strand == "-":
                temp_counts = temp_counts[::-1]
            total_counts += temp_counts
        total_sum = int(np.sum(total_counts))
        assert total_overlap == total_sum, f"Overlap: {total_overlap} vs. Calculated overlap {total_sum}."

        if enrichment:
            total_counts = total_counts / np.mean(total_counts)
            name = "Enrichment"
        else:
            name = "Occurrences"

        if return_frame:
            total_counts = pl.Series(total_counts)\
                                .to_frame(name=name)
        elif not return_array:
            total_counts = list(total_counts)
        return total_counts

    def extract_PWM(self, 
                    intersect_df: pl.DataFrame, 
                    window_size: int, 
                    return_frame: bool = True
                    ) -> dict[str, list[int]]:
        """
        Extracts position weight matrix (PWM) counts for each nucleotide at each position.

        Args:
            intersect_df (pl.DataFrame): DataFrame of intersected motif data.
            window_size (int): Window size for PWM extraction.
            return_frame (bool, optional): Whether to return as DataFrame. Defaults to True.

        Returns:
            dict[str, list[int]]: Dictionary of PWM counts for each nucleotide.
        """
        total_counts = {n: [0 for _ in range(2*window_size+1)] for n in self.nucleotides}
        total_overlap = 0
        for row in intersect_df.iter_rows(named=True):
            compartment_strand = row['strand']
            if compartment_strand == "?":
                continue
            start = int(row['start'])
            end = int(row['end'])
            motif_start = int(row['motif_start'])
            motif_end = int(row['motif_end'])
            sequence = row['sequence']
            if compartment_strand == "-":
                sequence = "".join(PWMExtractor.invert(n) for n in sequence)[::-1]
            overlap = int(row['overlap'])
            total_overlap += overlap
            origin = end - window_size - 1
            L = max(0, window_size - (origin - motif_start))
            U = min(2 * window_size + 1, window_size - (origin - motif_end))

            assert L <= U
            overlap_start = max(motif_start, start)
            overlap_end = min(motif_end, end)
            overlap_length = overlap_end - overlap_start
            assert overlap == overlap_length

            overlapping_sequence = sequence[max(0, start-motif_start): min(end-motif_start, len(sequence))]
            assert len(overlapping_sequence) == overlap == U-L

            for idx, pos in enumerate(range(L, U)):
                if compartment_strand == "-":
                    index = 2 * window_size - pos
                else:
                    index = pos
                nucl = overlapping_sequence[idx]
                total_counts[nucl][index] += 1

        assert total_overlap == sum(sum(v) for v in total_counts.values())
        return total_counts

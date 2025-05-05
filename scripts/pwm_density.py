import numpy as np
import pandas as pd
import polars as pl
from typing import Callable, Optional, Iterator
from abc import abstractmethod 
from tqdm import tqdm
import matplotlib.pyplot as plt
from seaborn import color_palette

class PWMExtractor:

    def __init__(self) -> None:
        self.nucleotides = "agct"

    @staticmethod
    def invert(nucleotide: str) -> str:
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
        return (counts+1)/(total_counts+total_bins)

    @staticmethod
    def expected_value(counts: int, total_counts: int, total_bins: int) -> float:
        return PWMExtractor.bayes_estimator(counts, total_counts, total_bins)

    @staticmethod
    def enrichment(counts: int, total_counts: int, total_bins: int) -> float:
        return counts/(total_counts * PWMExtractor.bayes_estimator(counts, total_counts, total_bins))

    def get_relative_positions(self, 
                               intersect_df: pl.DataFrame, 
                               window_size: int) -> Iterator[np.ndarray]:
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

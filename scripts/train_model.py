from pathlib import Path
import polars as pl
import pandas as pd
from termcolor import colored
import joblib
import json
import numpy as np
from utils import parse_fasta
from tqdm import tqdm
from pybedtools import BedTool
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import percentileofscore
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from typing import Optional
from attr import field
import attr

def count_cpgi(sequence: str) -> int:
    total_cpg = 0
    for i in range(len(sequence)-1):
        chunk = sequence[i:i+2]
        if chunk == "cg":
            total_cpg += 1
    return total_cpg

def create_windows(sequence_sizes: str, total_bins=2000):
    windows = BedTool().window_maker(g=sequence_sizes, n=total_bins)
    regions = []
    for window in windows:
        regions.append({"seqID": window.chrom,
                        "start": window.start,
                        "end": window.stop
                       })
    regions = pl.DataFrame(regions)
    return regions

def extract_region_gc_content(df: pl.DataFrame, fasta_sequence: str) -> pl.DataFrame:
    regions_bed = BedTool.from_dataframe(df.to_pandas()).sort()
    seqfa = regions_bed.sequence(fi=fasta_sequence, name=True, tab=True)
    df_with_gc = []
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
            df_with_gc.append(
                        {
                            "seqID": seqID,
                             "start": start,
                             "end": end,
                             "allele": allele,
                             "sequence": sequence,
                             "gc_proportion": gc_proportion,
                             "gc_content": gc_content,
                             "sequence_length": sequence_length
                        }
                    )
    df_with_gc = pl.DataFrame(df_with_gc)
    return df_with_gc

def split_genomic_regions(fasta_sequence: str, CHUNK_SIZE: int) -> pl.DataFrame:
    region_df = []
    for seqID, seq in tqdm(parse_fasta(fasta_sequence), total=24):
        seq = seq.lower()
        sequence_length = len(seq)
        for i in range(0, sequence_length, CHUNK_SIZE):
            chunk = seq[i:i+CHUNK_SIZE]
            gc_content = chunk.count("g") + chunk.count("c")
            cpgi = count_cpgi(chunk)
            chunk_size = min(sequence_length - i, CHUNK_SIZE)
            region_df.append({
                            "seqID": seqID,
                            "start": i,
                            "end": min(sequence_length, i+CHUNK_SIZE),
                            "length": chunk_size,
                            "gc_content": gc_content,
                            "gc_proportion": gc_content / chunk_size,
                            "cpg": cpgi,
                            "cpg_proportion": cpgi / (chunk_size - 1),
            })
    region_df = pl.DataFrame(region_df)
    return region_df

def extract_motif_coverage(region_df: pl.DataFrame, motif_bed, genome_density: float) -> pl.DataFrame:
    if isinstance(region_df, pl.DataFrame):
        region_df = region_df.to_pandas()
    region_bed = BedTool.from_dataframe(region_df).sort()
    COVERAGE_FIELDS = ["totalHits", "overlappingBp", "compartmentLength", "coverage"]
    region_df = pl.read_csv(
                    region_bed.coverage(motif_bed).fn,
                    has_header=False,
                    separator="\t",
                    new_columns=list(region_df.columns) + COVERAGE_FIELDS
            ).with_columns(
                    (pl.col("coverage") * 1e6).alias("Motif Density"),
            )\
            .with_columns(
                    (pl.col("Motif Density") / genome_density).alias("fold_enrichment")
            )\
            .filter(pl.col("fold_enrichment") > 0)
    return region_df

@attr.s
class GCEnrichmentModel:

    outdir: str = field(default="gc_enrichment_model")
    lower: int = field(converter=int, default=1)
    max_iter: int = field(converter=int, default=100)
    cv: int = field(converter=int, default=10)
    degree: int = field(converter=int, default=2)
    include_bias: bool = field(converter=bool, default=True)
    scoring: str = field(default="r2")
    step: float = field(converter=float, default=1e6)
    patience: int = field(converter=int, default=2)
    models_path: str = field(init=False)
    training_path: str = field(init=False)
    regions_path: str = field(init=False)
    residuals_path: str = field(init=False)
    ns_color: str = field(default="crimson")
    sig_color: str = field(default="green")

    def __attrs_post_init__(self):
        self.outdir = Path(self.outdir).resolve()
        self.regions_path = self.outdir.joinpath("regions")
        self.models_path = self.outdir.joinpath("models")
        self.residuals_path = self.outdir.joinpath("residuals")
        self.training_path = self.outdir.joinpath("training")
        self.plots_path = self.outdir.joinpath("performance_plots")
        self.regions_path.mkdir(exist_ok=True)
        self.models_path.mkdir(exist_ok=True)
        self.residuals_path.mkdir(exist_ok=True)
        self.training_path.mkdir(exist_ok=True)
        self.plots_path.mkdir(exist_ok=True)
        self.cv_results = self.training_path.joinpath(f"cv_{self.cv}_results_degree_{self.degree}_bias_{self.include_bias}.json")
        self.features = ["gc_proportion"]

    def train(self, motif_bed, fasta: str, sequence_sizes: str):
        print("Initializing GC-model training...") 
        # construct regions
        cv_results = self.grid_search(motif_bed, fasta, sequence_sizes)

        # train model using best parameters
        self.train_best_model(cv_results)
        print(colored("Exiting...", "green"))

    def load_genome_density(self) -> dict:
        target = Path(f"{self.outdir}/genome_wide_densities.json")
        if target.is_file():
            with open(target, mode="r", encoding="utf-8") as f:
                return json.load(f)

    def calculate_genome_wide_density(self, motif_bed, sequence_sizes: str, override: bool = False):
        target = Path(f"{self.outdir}/genome_wide_densities.json")
        if target.is_file():
            with open(target, mode="r", encoding="utf-8") as f:
                if not override:
                    return json.load(f)
        genome_sizes = pl.read_csv(sequence_sizes, 
                                   separator="\t", 
                                   has_header=False,
                                   new_columns=["seqID", "size"])
        motif_densities = pl.read_csv(
                                motif_bed.sort().merge().sort().fn, 
                                has_header=False, 
                                separator="\t",
                                new_columns=["seqID", "start", "end"]
                    )\
                    .with_columns((pl.col("end") - pl.col("start")).alias("motif_length"))\
                    .group_by("seqID")\
                    .agg(
                            pl.col("motif_length").sum()
                    )\
                    .join(
                            genome_sizes,
                            how="left",
                            on="seqID"
                    )\
                    .with_columns(
                            (1e6 * pl.col("motif_length") / pl.col("size")).alias("motif_density")
                    )
        motif_densities.write_csv(f"{self.outdir}/sequence_densities.csv")
        genome_size = genome_sizes["size"].sum()
        total_motif_density = 1e6 * motif_densities["motif_length"].sum() / genome_size
        avg_motif_density = motif_densities["motif_density"].mean()
        std_motif_density = motif_densities["motif_density"].std()
        with open(target, mode="w", encoding="utf-8") as f:
            json.dump({
                       "genome_size": genome_size,
                       "motif_density": total_motif_density,
                       "avg_motif_density": avg_motif_density,
                       "std_motif_density": std_motif_density
                       }, f, indent=4)
        return self.calculate_genome_wide_density(motif_bed, sequence_sizes)

    def grid_search(self, motif_bed, fasta, sequence_sizes) -> dict:
        genome_wide_stats = self.calculate_genome_wide_density(motif_bed, sequence_sizes)
        genome_wide_density = genome_wide_stats["motif_density"]
        avg_scores = dict()
        CHUNK_SIZE = self.lower
        prev_score = None
        total_iterations = 0
        diminished = 0
        for total_iterations, CHUNK_SIZE in enumerate(range(self.lower, self.lower + self.max_iter, 1), 1):
            CHUNK_SIZE_ = int(CHUNK_SIZE * self.step)
            regions_df = split_genomic_regions(fasta, CHUNK_SIZE_)
            regions_df = extract_motif_coverage(regions_df, motif_bed, genome_wide_density)
            regions_df.write_csv(
                        self.regions_path.joinpath(f"regions_chunk_size_{CHUNK_SIZE}.txt"),
                        separator="\t"
                    )
            
            # cross validate
            regions_df = regions_df.to_pandas()
            X = regions_df[self.features]
            trans = PolynomialFeatures(degree=self.degree, 
                                       include_bias=self.include_bias)
            X_poly = trans.fit_transform(X)
            y = regions_df["fold_enrichment"]
            model = LinearRegression()
            cv_results = cross_validate(model, 
                                        X_poly, 
                                        y, 
                                        cv=self.cv,
                                        scoring=self.scoring, 
                                        n_jobs=-1)

            # fit and save actual model
            pred_pipe = make_pipeline(
                                PolynomialFeatures(degree=self.degree,
                                                   include_bias=self.include_bias),
                                LinearRegression()
                                )
            pred_pipe.fit(X, y)

            # store residuals
            y_pred = pred_pipe.predict(X)
            residuals = y - y_pred
            #  save trained model
            model_dest = self.models_path.joinpath(f"linreg_model_degree_{self.degree}_CHUNK_{CHUNK_SIZE}.pkl")
            with open(model_dest, mode="wb") as f:
                joblib.dump(pred_pipe, f)

            # save residuals of that model
            residuals_path = self.residuals_path\
                            .joinpath(f"residuals_chunk_{CHUNK_SIZE}_degree_{self.degree}_bias_{self.include_bias}.txt")
            residuals.to_frame(name="Residuals").to_csv(residuals_path, 
                                              index=False, 
                                              mode="w", 
                                              header=False)

            avg_performance = cv_results["test_score"].mean()
            avg_scores[CHUNK_SIZE] = avg_performance
            if prev_score is not None:
                if prev_score > avg_performance:
                    diminished += 1
                    if diminished > self.patience:
                        print(colored(f"Found best model after {total_iterations} total iterations!", "green"))
                        break
            prev_score = CHUNK_SIZE
            
        # save results
        with open(self.cv_results, mode="w", encoding="utf-8") as f:
            json.dump(avg_scores, f, indent=4)
        print(colored("Cross validation has finished!", "green"))
        return avg_scores

    def load_regions(self) -> dict[int, pl.DataFrame]:
        regions = dict()
        regions = {file.name.split(".txt")[0].split("_")[-1]: file for file in self.regions_path.glob("*.txt")}
        for CHUNK_SIZE in regions:
            region_df = pl.read_csv(regions[CHUNK_SIZE], separator="\t")
            regions.update({CHUNK_SIZE: region_df})
        print(colored(f"Loaded total `{len(regions)}` regions.", "green"))
        return regions

    def load_region(self, chunk_size: int) -> pl.DataFrame:
        target = self.regions_path.joinpath(f"regions_chunk_size_{chunk_size}.txt")
        return pl.read_csv(target, separator="\t")

    @staticmethod
    def _choose_best_model(cv_results: dict[str, float]) -> tuple:
        cv_results = sorted(cv_results.items(), 
                            key=lambda score: score[1], 
                            reverse=True)
        best_model = cv_results[0]
        print(colored(f"Best model {best_model}.", "green"))
        return best_model

    def retrieve_best(self) -> int:
        with open(self.cv_results, mode="r", encoding="utf-8") as f:
            cv_results = json.load(f)
        return GCEnrichmentModel._choose_best_model(cv_results)
        
    def train_best_model(self, cv_results: dict[str, float]):
        '''
        Trains best model from cross validation
        '''
        best_model = GCEnrichmentModel._choose_best_model(cv_results)
        best_chunk = best_model[0]
        print(colored(f"Best model was found to be {best_chunk} with an r2 score {best_model[1]:.2f}.", "green"))
        model_dest = self.models_path.joinpath(f"best_model_degree_{self.degree}_CHUNK_{best_chunk}.pkl")

        regions_df = self.load_region(best_chunk)
        if isinstance(regions_df, pl.DataFrame):
            regions_df = regions_df.to_pandas()

        X = regions_df[self.features]
        y = regions_df["fold_enrichment"]
        pred_pipe = make_pipeline(
                            PolynomialFeatures(degree=self.degree,
                                               include_bias=self.include_bias),
                            LinearRegression()
                            )
        pred_pipe.fit(X, y)

        # store residuals
        y_pred = pred_pipe.predict(X)
        residuals = y - y_pred

        with open(model_dest, mode="wb") as f:
            joblib.dump(pred_pipe, f)
        print(colored(f"Saved best model to disk!", "green"))
        
        residuals_path = self.residuals_path.joinpath(f"residuals_chunk_{best_chunk}_degree_{self.degree}_bias_{self.include_bias}.txt")
        residuals.to_frame(name="Residuals").to_csv(residuals_path, 
                                          index=False, 
                                          mode="w", 
                                          header=False)

    def load_residuals(self, chunk_size: int, degree: int, include_bias: bool = True) -> list[float]:
        residuals_path = self.residuals_path.joinpath(f"residuals_chunk_{chunk_size}_degree_{degree}_bias_{include_bias}.txt")
        residuals = []
        with open(residuals_path, mode="r", encoding="utf-8") as f:
            for line in f:
                residuals.append(float(line.strip()))
        return residuals

    def load_model(self, chunk_size: int, degree: int = 2):
        target = self.models_path.joinpath(f"best_model_degree_{degree}_CHUNK_{chunk_size}.pkl")
        with open(target, mode="rb") as f:
            return joblib.load(f)

    def adjust_gc(self, df: pd.DataFrame, chunk_size: int, degree: int) -> pl.DataFrame:
                   
        '''
        Input: 
        df: dataframe to perform GC adjustment on Fold Enrichment
        -------------------------
        - Loads model
        - Performs GC adjustment
        -------------------------
        Returns: GC-adjusted dataframe for motif density
        '''

        loaded_pipe = self.load_model(chunk_size, degree=degree)
        if "gc_proportion" not in df:
            raise  KeyError(f"GC-Content has not been calculated.")
        if isinstance(df, pl.DataFrame):
            df = df.to_pandas()

        X_test = df[self.features]
        y_pred = pl.Series(loaded_pipe.predict(X_test)).to_frame("predicted_enrichment")
        df_enrichment = pl.concat([
                                            df, y_pred
                                        ], how="horizontal")\
                                    .with_columns(
                                                (pl.col("fold_enrichment") - pl.col("predicted_enrichment")).alias("res")
                                    )\
                                    .with_columns(
                                                pl.col("res").map_elements(perform_test).alias("test_statistic")
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
                                    pl.col("adj_pval").map_elements(evaluate_stars, return_dtype=str).alias("significance")
                            )\
                            .with_columns(
                                pl.col("significance").map_elements(lambda x: self.sig_color if x.startswith("*") else self.ns_color, return_dtype=str).alias("significance_color")
                            )
        return df_enrichment 

    def plot_best_model(self, chunk_size: int, degree: int = 2, include_bias: bool = True) -> None:
        best_model = self.load_model(chunk_size, degree=degree)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

        # coef = best_model.coef_
        # predict = lambda arr: coef[0] + xs * coef[1] + coef[2] * (xs ** 2)
        region_df = model.load_region(chunk_size)
        X = region_df[self.features]
        y = region_df["fold_enrichment"]
        xs = np.linspace(0, 1.0, 200)
        # X_poly_test = trans.transform(xs.reshape(-1, 1))

        ax.plot(xs, 
                best_model.predict(X), 
                lw=2.0, 
                linestyle='--', 
                color='black', 
                label=f"Polynomial Regression\nWindow={chunk_size}Mb")
        ax.scatter(X, y, alpha=0.6, color='crimson')
        ax.set_xlim(0.15, 0.6)
        ax.set_ylim(ymin=-0.5, ymax=14)
        ax.grid(lw=0.4, alpha=0.6, zorder=0)
        ax.legend(prop={"size": 15}, loc="upper left")
        ax.tick_params(axis="both", labelsize=14)
        ax.xaxis.label.set_size(16)
        ax.set_axisbelow(True)
        ax.yaxis.label.set_size(16)
        ax.set_xlabel("GC Proportion")
        ax.set_ylabel("G4 Enrichment")
        fig.savefig(f"{self.plots_path}/model_pred_chunk_{chunk_size}_degree_{degree}_linreg.png", bbox_inches='tight')
        
        # plot residuals
        residuals = self.load_residuals(chunk_size=chunk_size, degree=degree, include_bias=include_bias)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
        sns.histplot(residuals, ax=ax, bins=40)
        ax.grid(lw=0.4, alpha=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylabel("Count")
        ax.set_xlabel("Residual ε")
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)
        ax.tick_params(axis="both", labelsize=14)
        ax.set_xlim(xmin=-1.5, xmax=1.5)
        fig.savefig(f"{self.plots_path}/model_residuals_chunk_{chunk_size}_degree_{degree}_linreg.png", bbox_inches='tight')
        # res.to_frame(name="Residuals").to_csv("/storage/group/izg5139/default/nicole/g4_t2t_analysis/datasets/residuals_2degree_g4_model.csv", index=False, mode="w", header=False)

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
    

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description="""GC Model trainer.""")
    parser.add_argument("motifs", type=str)
    parser.add_argument("fasta", type=str)
    parser.add_argument("--sequence_sizes", type=str)
    parser.add_argument("--lower", type=int, default=1)
    parser.add_argument("--cv", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--include_bias", type=int, default=1, choices=[0, 1])
    parser.add_argument("--step", type=float, default=1e6)
    parser.add_argument("--scoring", type=str, default="r2")
    parser.add_argument("--sig_color", type=str, default="green")
    parser.add_argument("--ns_color", type=str, default="crimson")
    args = parser.parse_args()
    
    outdir = Path("model_training")
    outdir.mkdir(exist_ok=True)

    motifs_df = pd.read_table(args.motifs, usecols=range(3))
    motif_bed = BedTool.from_dataframe(motifs_df).sort()
    
    model = GCEnrichmentModel(outdir=outdir,
                              lower=args.lower,
                              cv=args.cv,
                              degree=args.degree,
                              include_bias=args.include_bias,
                              scoring=args.scoring,
                              step=args.step,
                              patience=args.patience,
                              sig_color=args.sig_color,
                              ns_color=args.ns_color
                              )
    model.train(motif_bed, args.fasta, args.sequence_sizes)

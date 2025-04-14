from pathlib import Path
from scipy.stats import percentileofscore
from statsmodels.stats.multitest import multipletests
import pandas as pd
import polars as pl
import numpy as np
from constants import ConfigPaths
import joblib
import attr
from attrs import field

@attr.s
class HypothesisTest:

    pvalue: float = field()
    stat: float = field()

class GCAdjustmentResidualTest:

    def __init__(self, threshold: float = 1.6, degree: int = 2, CHUNK_SIZE: int = 2):
        self.threshold = threshold
        self.degree = degree
        self.CHUNK_SIZE = CHUNK_SIZE
        self.model = self.load_model()
        self.model_residuals = self.load_residuals()

    def residual_test(self, observed_residual, alternative = "one-tailed") -> HypothesisTest:
        if alternative != "one-tailed" and alternative != "two_tailed":
            raise ValueError("Alternative hypothesis must be either `one-tailed` or `two-tailed`.")
        percentile = percentileofscore(self.model_residuals, 
                                       observed_residual, 
                                       kind='rank')
        pval = 1 - percentile/1e2 if observed_residual > np.median(self.model_residuals) else percentile/1e2
        return HypothesisTest(stat=percentile, 
                              pvalue=pval) \
                if alternative == "one-tailed" \
                else HypothesisTest(stat=percentile, 
                                    pvalue=pval*2)

    def load_model(self):
        model = Path(f"/storage/group/izg5139/default/nicole/g4_t2t/src/scripts/model_training/models/best_model_degree_{self.degree}_CHUNK_{self.CHUNK_SIZE}.pkl")
        with open(model, 'rb') as f:
            linreg = joblib.load(f)
        return linreg
    
    def evaluate_stars(self, pval: float) -> str:
        if pval < 0.001:
            return "*" * 3
        if pval < 0.01:
            return "*" * 2
        if pval < 0.05:
            return "*"
        return "ns"
    
    def load_residuals(self) -> np.ndarray:
        residuals = []
        with open("/storage/group/izg5139/default/nicole/g4_t2t/src/scripts/model_training/residuals/residuals_chunk_2_degree_2_bias_True.txt") as f:
            for line in f:
                residuals.append(float(line.strip()))
        residuals = np.array(residuals)
        residuals = residuals[np.abs(residuals) <= self.threshold]
        return residuals

def load_gw_density(g4_bed, g4_control_bed):
    # Genome Size
    genome_size = pd.read_table(ConfigPaths.GENOME_SIZE.value, 
                                header=None, 
                                names=["seqID", "size"])["size"].sum()

    # G4 DENSITY
    g4_gw_density = pd.read_table(
                                g4_bed.sort().merge().fn, 
                                header=None, 
                                names=["seqID", "start", "end"]
                    )
    g4_gw_density["size"] = g4_gw_density["end"] - g4_gw_density["start"]
    g4_gw_density = g4_gw_density["size"].sum() * 1e6 / genome_size

    # CONTROL DENSITY
    control_gw_density = pd.read_table(
                                g4_control_bed.sort().merge().fn, header=None, names=["seqID", "start", "end"]
    )
    control_gw_density["size"] = control_gw_density["end"] - control_gw_density["start"]
    control_gw_density = control_gw_density["size"].sum() * 1e6 / genome_size

    return g4_gw_density, control_gw_density

def adjust_for_gc_content(df: pl.DataFrame, degree: int = 2, CHUNK_SIZE: int = 2):
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    def perform_test(res: float) -> tuple[float, float]:
        hypothesis= residual_test.residual_test(res)
        return hypothesis.stat, hypothesis.pvalue

    residual_test = GCAdjustmentResidualTest(threshold=1.6, 
                                             degree=degree, 
                                             CHUNK_SIZE=CHUNK_SIZE)
    linreg = residual_test.load_model()
    y_pred = pl.Series(linreg.predict(df.to_pandas()[["gc_proportion"]])).to_frame(name='predicted_enrichment')
    df_adj = pl.concat([
                        df, 
                        y_pred
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
    p_values = list(df_adj["pval"])
    corrected_pvals = multipletests(p_values, method='fdr_bh')[1]
    df_adj = pl.concat([
                        df_adj,
                        pl.Series(corrected_pvals).to_frame("adj_pval"),
                    ], how="horizontal"
                )\
                .with_columns(
                        pl.col("adj_pval").map_elements(residual_test.evaluate_stars, return_dtype=str).alias("significance")
                ) # \
                        # .with_columns(
                        #             pl.col("adj_pval").map_elements(lambda x: -math.log(x, 10), return_dtype=float).alias("-log(pval_adj)"),
                        #             pl.col("pval").map_elements(lambda x: -math.log(x, 10), return_dtype=float).alias("-log(pval)")

                        # )
    return df_adj

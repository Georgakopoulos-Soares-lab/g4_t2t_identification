from pathlib import Path
import math
from scipy.stats import percentileofscore, chi2_contingency
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
    """
    Stores the result of a hypothesis test.

    Args:
        pvalue (float): The p-value of the test.
        stat (float): The test statistic (e.g., percentile).

    Returns:
        HypothesisTest: An object containing the p-value and statistic.
    """

    pvalue: float = field()
    stat: float = field()

class ContigencyTest:
    """
    Performs a chi-squared contingency test on a 2x2 table.
    """
    def contingency(self, row) -> float:
        """
        Computes the chi-squared test for independence for a given row.

        Args:
            row (dict): Dictionary with keys 'overlappingBp', 'not_g4', 'overlappingBp_control', 'not_control'.

        Returns:
            tuple: The result of scipy.stats.chi2_contingency (statistic, p-value, dof, expected).
        """
        array = np.array([
                        [row["overlappingBp"], row["not_g4"]],
                        [row["overlappingBp_control"], row["not_control"]]
                        ],
                        dtype=np.int64
                        )
        return chi2_contingency(array)

class GCAdjustmentResidualTest:
    """
    Performs GC content adjustment and residual hypothesis testing using a pre-trained model.
    """
    def __init__(self, model_path: str, threshold: float = 1.6, degree: int = 2, CHUNK_SIZE: int = 2):
        """
        Initializes the GCAdjustmentResidualTest.

        Args:
            model_path (str): Path to the model directory.
            threshold (float, optional): Residual threshold for filtering. Defaults to 1.6.
            degree (int, optional): Degree of the polynomial model. Defaults to 2.
            CHUNK_SIZE (int, optional): Chunk size for model. Defaults to 2.
        """
        self.model_path = model_path
        self.threshold = threshold
        self.degree = degree
        self.CHUNK_SIZE = CHUNK_SIZE
        self.model = self.load_model()
        self.model_residuals = self.load_residuals()

    def residual_test(self, observed_residual, alternative = "one-tailed") -> HypothesisTest:
        """
        Performs a residual hypothesis test comparing the observed residual to the model residuals.

        Args:
            observed_residual (float): The observed residual value.
            alternative (str, optional): 'one-tailed' or 'two-tailed'. Defaults to 'one-tailed'.

        Returns:
            HypothesisTest: Object containing the test statistic and p-value.
        """
        if alternative != "one-tailed" and alternative != "two-tailed":
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
        """
        Loads the linear regression model from disk.

        Args:
            None

        Returns:
            sklearn.linear_model: The loaded regression model.
        """
        model = Path(f"{self.model_path}/models/linreg_model_degree_{self.degree}_CHUNK_{self.CHUNK_SIZE}.pkl")
        with open(model, 'rb') as f:
            linreg = joblib.load(f)
        return linreg
    
    @staticmethod
    def evaluate_stars(pval: float) -> str:
        """
        Maps a p-value to a significance string (stars or 'ns').

        Args:
            pval (float): The p-value to evaluate.

        Returns:
            str: Significance as stars or 'ns'.
        """
        if pval < 0.0001:
            return "*" * 4
        if pval < 0.001:
            return "*" * 3
        if pval < 0.01:
            return "*" * 2
        if pval < 0.05:
            return "*"
        return "ns"
    
    def load_residuals(self) -> np.ndarray:
        """
        Loads and filters model residuals from disk.

        Args:
            None

        Returns:
            np.ndarray: Array of filtered residuals.
        """
        residuals = []
        with open(f"{self.model_path}/residuals/residuals_chunk_2_degree_2_bias_True.txt") as f:
            for line in f:
                residuals.append(float(line.strip()))
        residuals = np.array(residuals)
        residuals = residuals[np.abs(residuals) <= self.threshold]
        return residuals

def load_gw_density(g4_bed, g4_control_bed) -> tuple[float, float]:
    """
    Calculates genome-wide density for G4 and control motifs.

    Args:
        g4_bed (BedTool): BedTool object for G4 motifs.
        g4_control_bed (BedTool): BedTool object for control motifs.

    Returns:
        tuple[float, float]: (G4 density, control density) per Mb.
    """
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
                                g4_control_bed.sort().merge().fn, 
                                header=None, 
                                names=["seqID", "start", "end"]
    )
    control_gw_density["size"] = control_gw_density["end"] - control_gw_density["start"]
    control_gw_density = control_gw_density["size"].sum() * 1e6 / genome_size

    return g4_gw_density, control_gw_density


def adjust_for_gc_content(df: pl.DataFrame, 
                          model_path: str,
                          degree: int = 2, 
                          CHUNK_SIZE: int = 2,
                          threshold: float = 1.6,
                          alternative: str = 'two-tailed',
                          evaluate_stars: bool = True) -> pl.DataFrame:
    """
    Adjusts fold enrichment for GC content using a regression model and computes residual significance.

    Args:
        df (pl.DataFrame): DataFrame with 'fold_enrichment' and 'gc_proportion'.
        model_path (str): Path to the model directory.
        degree (int, optional): Degree of the polynomial model. Defaults to 2.
        CHUNK_SIZE (int, optional): Chunk size for model. Defaults to 2.
        threshold (float, optional): Residual threshold for filtering. Defaults to 1.6.
        alternative (str, optional): 'one-tailed' or 'two-tailed'. Defaults to 'two-tailed'.
        evaluate_stars (bool, optional): Whether to add significance stars. Defaults to True.

    Returns:
        pl.DataFrame: DataFrame with predicted enrichment, residuals, p-values, and significance.
    """
    
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)

    def perform_test(res: float, alternative: str) -> tuple[float, float]:
        hypothesis= residual_test.residual_test(res, alternative=alternative)
        return hypothesis.stat, hypothesis.pvalue

    residual_test = GCAdjustmentResidualTest(threshold=threshold, 
                                             model_path=model_path,
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
                        pl.col("res").map_elements(lambda res: perform_test(res, alternative=alternative)).alias("test_statistic")
                )\
                .with_columns(
                        pl.col("test_statistic").list.get(0).alias("percentile"),
                        pl.col("test_statistic").list.get(1).alias("pval")
            ).drop(['test_statistic'])
    if evaluate_stars:
        df_adj = df_adj.with_columns(
                        pl.col("pval").map_elements(GCAdjustmentResidualTest.evaluate_stars, return_dtype=str).alias("gc_stars")
                        )

    return df_adj

def correct_multiple_comparisons(df_adj: pl.DataFrame, 
                                strategy: str = "fdr_bh",
                                use_log: bool = True) -> pl.DataFrame:
    """
    Applies multiple testing correction to p-values and adds adjusted significance columns.

    Args:
        df_adj (pl.DataFrame): DataFrame with 'pval' column.
        strategy (str, optional): Correction method (default 'fdr_bh').
        use_log (bool, optional): Whether to add -log(pval) columns. Defaults to True.

    Returns:
        pl.DataFrame: DataFrame with adjusted p-values and significance columns.
    """
    p_values = list(df_adj["pval"])
    corrected_pvals = multipletests(p_values, method='fdr_bh')[1]
    df_adj = pl.concat([
                       df_adj,
                        pl.Series(corrected_pvals).to_frame("adj_pval"),
                    ], how="horizontal"
                )\
                .with_columns(
                        pl.col("adj_pval").map_elements(GCAdjustmentResidualTest.evaluate_stars, return_dtype=str).alias("adj_significance")
                )
    if use_log:
        df_adj = df_adj.with_columns(
                        pl.col("adj_pval").map_elements(lambda x: -math.log(x, 10), return_dtype=float).alias("-log(pval_adj)"),
                        pl.col("pval").map_elements(lambda x: -math.log(x, 10), return_dtype=float).alias("-log(pval)")
                    )
    return df_adj
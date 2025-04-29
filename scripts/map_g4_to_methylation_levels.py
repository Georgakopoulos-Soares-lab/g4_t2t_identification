from attr import field
from pathlib import Path
import attr
from tqdm import tqdm
from termcolor import colored
import polars as pl
from pybedtools import BedTool
from enum import Enum
import re
from constants import ConfigPaths

def extract_grun_counts(sequence: str) -> int:
    motif = "g" if sequence.count("g") >= sequence.count("c") else "c"
    total_gruns = list(filter(len, re.findall("%s{3,}" % motif, sequence)))
    return len(total_gruns)

def extract_loop_ratio(sequence: str) -> float:
    motif = "g" if sequence.count("g") >= sequence.count("c") else "c"
    loop_length = len(re.sub(r"%s{3,}" % motif, "", sequence))
    return 1e2 * loop_length / float(len(sequence))

@attr.s
class MethylationMapper:
    
    methylation_type: str = field()
    methylation_path: str = field(init=False)
    g4_type: str = field()
    g4_path: str = field(init=False)
    g4_controls: str = field(init=False)

    out: str = field(default=None)
    methylation_levels: list[str] = field(init=False)

    def __attrs_post_init__(self) -> None:
        if self.out:
            self.out = Path(self.out).resolve()
            self.out.mkdir(exist_ok=True)
            print(colored(f"Created output directory `{self.out}`.", "green"))
        else:
            print(colored(f"Failed to create output directory.", "red"))

        if self.methylation_type == "HG002_METH":
            self.methylation_path = ConfigPaths.HG002_METH.value
        elif self.methylation_type == "CHM13v2_METH":
            self.methylation_path = ConfigPaths.CHM13v2_METH.value
        else:
            raise ValueError(f"Invalid methylation type `{self.methylation_type}`.")

        if self.g4_type == "G4HUNTER":
            self.g4_path = ConfigPaths.G4HUNTER.value
            self.g4_controls = ConfigPaths.CONTROL_G4HUNTER.value
        elif self.g4_type == "G4REGEX":
            self.g4_path = ConfigPaths.G4REGEX.value
            self.g4_controls = ConfigPaths.CONTROL_G4REGEX.value
        else:
            raise ValueError(f"Invalid G4 type `{self.g4_type}`.")

        self.methylation_path = Path(self.methylation_path).resolve()
        self.g4_path = Path(self.g4_path).resolve()
        self.g4_controls = Path(self.g4_controls).resolve()

        self.methylation_levels: list[str] = ["Hypomethylated", "Methylated", "Hypermethylated"]
        return

    def map_methylation(self, df: pl.DataFrame, methylation_bed = None) -> pl.DataFrame:
        if isinstance(df, pl.DataFrame):
            df_bed = BedTool.from_dataframe(df.to_pandas()).sort()
        else:
            df_bed = BedTool.from_dataframe(df).sort()

        if methylation_bed is None:
            methylation_bed = BedTool.from_dataframe(self.load_methylation().to_pandas()).sort()
        df_methylated = pl.read_csv(
                        df_bed.intersect(methylation_bed, wo=True).fn,
                        has_header=False,
                        separator="\t",
                        new_columns=list(df.columns) \
                                    + ["chromosome", "methylation_start", "methylation_end", "methylation_level", "overlap"]
            )\
            .group_by(list(df.columns), maintain_order=True)\
            .agg(
                    pl.col("methylation_level").mean().alias("avg_methylation"),
                    pl.col("overlap").sum().alias("methylation_hits"),
                    pl.col("overlap").count().alias("methylation_counts")
            )\
            .with_columns(
                    pl.col("avg_methylation").map_elements(MethylationMapper.extract_methylation, return_dtype=str).alias("methylation_level")
            )
        return df_methylated

    def load_methylation(self) -> pl.DataFrame:
        methylation_df = pl.read_csv(
                                 self.methylation_path,
                                 separator="\t", 
                                 has_header=False,
                                 new_columns=["seqID", "start", "end", "methylation_level"]
                                )\
                    .filter(
                                pl.col("start") < pl.col("end")
                                )
        return methylation_df

    def load_g4hunter(self) -> pl.DataFrame:
        g4_df = pl.read_csv(
                    self.g4_path,
                    separator="\t",
                    columns=["seqID", "start", "end", "sequence", "score"]
                    )\
                .with_columns(
                        pl.col("sequence").str.to_lowercase()
                )\
                .with_columns(
                            pl.col("sequence")
                                            .map_elements(lambda seq: "+" if seq.count("g") >= seq.count("c") else "-",
                                                                               return_dtype=str)
                                            .alias("motif_strand"),

                            pl.col("sequence").map_elements(extract_grun_counts, return_dtype=int).alias("GRUN_counts"),

                            pl.col("sequence").map_elements(extract_loop_ratio, return_dtype=float).alias("LOOP_ratio"),
                    )
        return g4_df

    def load_g4_regex(self) -> pl.DataFrame:
        g4_df = pl.read_csv(self.g4_path,
                            separator="\t"
                            )\
                    .with_columns(
                                pl.col("sequence").str.to_lowercase()
                    )\
                    .with_columns(
                            pl.col("sequence")
                                            .map_elements(lambda seq: "+" if seq.count("g") >= seq.count("c") else "-",
                                                                               return_dtype=str)
                                            .alias("motif_strand"),

                            pl.col("sequence").map_elements(extract_grun_counts, return_dtype=int).alias("GRUN_counts"),

                            pl.col("sequence").map_elements(extract_loop_ratio, return_dtype=float).alias("LOOP_ratio"),
                    )
        return g4_df
    
    def load_g4_controls(self) -> pl.DataFrame:
        g4_controls = pl.read_csv(self.g4_controls, separator="\t")
        return g4_controls

    @staticmethod
    def extract_methylation(meth_prob: float) -> str:
        if meth_prob < 0.2:
            return "Hypomethylated"
        if meth_prob < 0.7:
            return "Methylated"
        return "Hypermethylated"

    def map_g4_to_methylation(self) -> pl.DataFrame:
        # LOAD G4
        if self.g4_type == "G4HUNTER":
            g4_df = self.load_g4hunter()
            print("Loaded G4HUNTER!")
        else:
            g4_df = self.load_g4_regex()
            print("LOADED REGEX!")
        
        g4_bed = BedTool.from_dataframe(g4_df.to_pandas()).sort()
        
        # LOAD CONTROLS
        g4_controls = self.load_g4_controls()
        g4_controls_bed = BedTool.from_dataframe(g4_controls.to_pandas()).sort()

        # LOAD METHYLATION
        methylation_df = self.load_methylation()
        methylation_bed = BedTool.from_dataframe(methylation_df.to_pandas()).sort()

        g4_df_meth = pl.read_csv(
                    g4_bed.intersect(methylation_bed, wo=True).fn,
                    has_header=False,
                    separator="\t",
                    new_columns=g4_df.columns + ["chromosome", 
                                                 "meth_start", 
                                                 "meth_end", 
                                                 "methylation_level",
                                                 "overlap"]
                ).group_by(g4_df.columns, maintain_order=True)\
                        .agg(
                                pl.col("methylation_level").mean().alias("avg_methylation"),
                                pl.col("methylation_level").count().alias("methylation_count"),
                        ).with_columns(
                                pl.col("avg_methylation").map_elements(MethylationMapper.extract_methylation, return_dtype=str)\
                                        .alias("methylation_level")
                        )
        g4_df_meth.write_csv(f"{self.out}/chm13v2_{self.methylation_type}_{self.g4_type}.txt", separator="\t")


        # Save G4Hunter with enriched statistics
        g4_df.write_csv(f"{self.out}/chm13v2_g4hunter.enriched.txt", separator="\t")
        
        g4_controls_meth = pl.read_csv(                
                        g4_controls_bed.intersect(methylation_bed, wo=True).fn,
                        has_header=False,
                        separator="\t",
                        new_columns=g4_controls.columns + ["chromosome", 
                                                           "meth_start", 
                                                           "meth_end", 
                                                           "methylation_level",
                                                           "overlap"]
                    ).group_by(g4_controls.columns, maintain_order=True)\
                     .agg(
                                pl.col("methylation_level").mean().alias("avg_methylation"),
                                pl.col("methylation_level").count().alias("methylation_count"),
                        ).with_columns(
                                pl.col("avg_methylation").map_elements(MethylationMapper.extract_methylation, return_dtype=str)\
                                        .alias("methylation_level")
                            )

        g4_controls_meth.write_csv(f"{self.out}/chm13v2_{self.methylation_type}_{self.g4_type}.controls.txt", separator="\t")

        for methylation_level in tqdm(self.methylation_levels):
            g4_df_meth.filter(pl.col("methylation_level") == methylation_level)\
                      .write_csv(f"{self.out}/chm13v2_{self.methylation_type}_{self.g4_type}.{methylation_level}.txt",
                                        separator="\t")

            g4_controls_meth.filter(pl.col("methylation_level") == methylation_level)\
                      .write_csv(f"{self.out}/chm13v2_{self.methylation_type}_{self.g4_type}.{methylation_level}.controls.txt",
                                        separator="\t")
        return

if __name__ == "__main__":
    
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument("--outdir", type=str, default="/storage/group/izg5139/default/nicole/g4_t2t_identification/methylation_data")
   args = parser.parse_args()

   methylation_types = ["HG002_METH", "CHM13v2_METH"]
   g4_types = ["G4HUNTER", "G4REGEX"]

   for methylation_type in tqdm(methylation_types):
       for g4_type in tqdm(g4_types):
           mapper = MethylationMapper(methylation_type=methylation_type,
                                      g4_type=g4_type,
                                      out=args.outdir)
           mapper.map_g4_to_methylation()

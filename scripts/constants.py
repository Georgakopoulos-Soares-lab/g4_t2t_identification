from dotenv import load_dotenv
import pathlib
from pathlib import Path
from enum import Enum
import os
from typing import TypeVar

PathLike = TypeVar("PathLike", str, pathlib.Path, None)
load_dotenv()

class ConfigPaths(Enum):

    CENTROMERE = os.getenv("CENTROMERE")
    TELOMERE = os.getenv("TELOMERE")
    # SEQUENCE_SIZES = Path(os.getenv("SEQUENCE_SIZES"))
    GENOME_SIZE = os.getenv("GENOME_SIZE")
    INDEX = os.getenv("INDEX")
    FASTA = os.getenv("FASTA")

    # TRIMERS = Path(os.getenv("TRIMERS"))
    G4HUNTER = os.getenv("G4HUNTER")
    CONTROL_G4HUNTER = os.getenv("CONTROL_G4HUNTER")

    # METHYLATION DATASETS
    HG002_METH = os.getenv("HG002_METH")
    CHM13v2_METH = os.getenv("CHM13v2_METH")

    # METADATA
    METADATA = os.getenv("METADATA")

    # G4 DATASETS
    G4HUNTER_HG002 = os.getenv("G4HUNTER_HG002")
    G4REGEX = os.getenv("G4REGEX")
    CONTROL_G4REGEX = os.getenv("CONTROL_G4REGEX")

    # SEQUENCE REPORT
    SEQUENCE_REPORT = os.getenv("SEQUENCE_REPORT")

    # GFF
    GFF = os.getenv("GFF")
    GFF_AGAT = os.getenv("GFF_AGAT")

    # MISCALLENEOUS
    # PROMOTERS = Path(os.getenv("PROMOTERS"))
    GERM_VAR = os.getenv("GERM_VAR")
    MUTATION = os.getenv("MUTATION")

    # directory of processed/splitted mutations from Germline Variants
    PROCESSED_MUTATIONS = os.getenv("PROCESSED_MUTATIONS")
    PRMD9 = os.getenv("PRMD9")

    def __init__(self, path):
        if path and not Path(path).resolve().exists():
            raise FileNotFoundError(f"Path does not exist: {path}.")

if __name__ == "__main__":
    
    config = ConfigPaths


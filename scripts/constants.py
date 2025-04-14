from dotenv import load_dotenv
import pathlib
from pathlib import Path
from enum import Enum
import os
from typing import TypeVar

PathLike = TypeVar("PathLike", str, pathlib.Path, None)
load_dotenv()

class ConfigPaths(Enum):

    CENTROMERE = Path(os.getenv("CENTROMERE"))
    TELOMERE = Path(os.getenv("TELOMERE"))
    SEQUENCE_SIZES = Path(os.getenv("SEQUENCE_SIZES"))
    GENOME_SIZE = Path(os.getenv("GENOME_SIZE"))
    GENOME = Path(os.getenv("GENOME"))
    REPORT = Path(os.getenv("REPORT"))
    INDEX = Path(os.getenv("INDEX"))
    FASTA = Path(os.getenv("FASTA"))

    # TRIMERS = Path(os.getenv("TRIMERS"))
    G4HUNTER = Path(os.getenv("G4HUNTER"))
    CONTROL_G4HUNTER = Path(os.getenv("CONTROL_G4HUNTER"))

    # METHYLATION DATASETS
    HG002_METH = Path(os.getenv("HG002_METH"))
    CHM13v2_METH = Path(os.getenv("CHM13v2_METH"))

    # METADATA
    METADATA = Path(os.getenv("METADATA"))

    # G4 DATASETS
    G4HUNTER_HG002 = Path(os.getenv("G4HUNTER_HG002"))
    G4REGEX = Path(os.getenv("G4REGEX"))
    CONTROL_REGEX = ""

    # SEQUENCE REPORT
    SEQUENCE_REPORT = Path(os.getenv("SEQUENCE_REPORT"))

    # GFF
    GFF = Path(os.getenv("GFF"))
    GFF_AGAT = Path(os.getenv("GFF_AGAT"))

    # MISCALLENEOUS
    PROMOTERS = Path(os.getenv("PROMOTERS"))
    GERM_VAR = Path(os.getenv("GERM_VAR"))
    MUTATION = Path(os.getenv("MUTATION"))

    def __init__(self, path):
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}.")

if __name__ == "__main__":
    
    config = ConfigPaths


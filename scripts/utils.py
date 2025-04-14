from Bio.SeqIO.FastaIO import SimpleFastaParser
from pathlib import Path
import gzip
from typing import Iterator

def parse_fasta(file: str) -> Iterator[tuple[str, str]]:
    if Path(file).name.endswith(".gz"):
        f = gzip.open(file, "rt")
    else:
        f = open(file, mode="r", encoding="utf-8")
    for record in SimpleFastaParser(f):
        seqID = record[0]
        if " " in seqID:
            seqID = seqID.split(" ")[0]
        seq = record[1]
        yield seqID, seq
    f.close()

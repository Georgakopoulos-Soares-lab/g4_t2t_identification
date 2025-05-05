from utils import parse_fasta
from tqdm import tqdm
from pathlib import Path
import csv
import sys

def reverse_complement(kmer):
    return ''.join({'a': 't',
                    't': 'a',
                    'g': 'c',
                    'c': 'g'}[n] for n in kmer)[::-1]

if __name__ == "__main__":

    human_genome = "/storage/group/izg5139/default/nicole/g4_t2t_analysis/datasets/chm13v2.0.fa.gz"
    motif = sys.argv[1].strip().lower()
    reverse_motif = reverse_complement(motif)
    motif_length = len(motif)

    outdir = Path("motif_occurrences")
    outdir.mkdir(exist_ok=True)

    output = open(f"motif_counts_{motif}.bed", mode="w")
    writer = csv.DictWriter(output, fieldnames=["seqID", "start", "end", "motif"], delimiter="\t")

    for seqID, seq in parse_fasta(human_genome):
        seq = seq.lower()
        seq_length = len(seq)

        for i in tqdm(range(seq_length - motif_length + 1)):
            chunk = seq[i:i+motif_length]

            if chunk == motif or chunk == reverse_motif:
                writer.writerow({
                    "seqID": seqID,
                    "start": i,
                    "end": i + motif_length,
                    "motif": motif}
                    )
    output.close()

# Control Generator for sequence Sequences in Human Genome
import pandas as pd
from tqdm import tqdm
import gzip
import random
import threading
import time
import csv
import os
import argparse
import concurrent.futures
import multiprocessing
from pathlib import Path
import tempfile
import logging
from pybedtools import BedTool
from dotenv import load_dotenv
from Bio.SeqIO.FastaIO import SimpleFastaParser
from functools import reduce

loaded = load_dotenv()
if loaded:
    raise ValueError('Failed to load .env file.')

# os.environ['BEDTOOLS'] = os.getenv('bedtools')
# print(os.environ['BEDTOOLS'])
INTERSECT_FIELDS = ["seqID",
                    "start",
                    "end",
                    "sequence",
                    "controls_for",
                    "length",
                    "gc_content",
                    "cg_counts",
                    "gc_counts",
                    "chromosome",
                    "mStart",
                    "mEnd",
                    "overlap"]
# print(os.getenv('bedtools'))
# pybedtools.helpers.set_bedtools_path(os.getenv('bedtools'))
def parse_fasta(fasta: str):
    if Path(fasta).name.endswith(".gz"):
        file = gzip.open(fasta, 'rt')
    else:
        file = open(fasta, mode='r', encoding='utf-8')
    for record in SimpleFastaParser(file):
        yield record[0].split(' ')[0], record[1].strip().lower()
    file.close()

def find_steps(loc, window, seq_length):
    right_step = 1
    left_step = 1
    for i in range(seq_length):
        if i%2 == 0 and loc+right_step+window < seq_length:
            yield (loc+right_step, loc+right_step+window)
            right_step += 1
        elif loc >= left_step and loc-left_step+window < seq_length:
            yield (loc-left_step, loc-left_step+window)
            left_step += 1

def count_occurrences(seq: str) -> tuple[int, int]:
    counter_gc = 0
    counter_cg = 0
    for i in range(len(seq)-1):
        if seq[i:i+2] == "cg":
            counter_cg += 1
        elif seq[i:i+2] == "gc":
            counter_gc += 1
    return counter_cg, counter_gc

def load_sequences(sequence_path: str) -> list[dict]:
    all_sequences = []
    with open(sequence_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row['seqID'] == chromosome:
                row['sequence'] = row['sequence'].lower()
                all_sequences.append(row)
    return all_sequences

def search_chromosome(fasta: str, chromosome: str) -> tuple[str, str]:
    print(f"Infile <--- {fasta}.")
    gen = parse_fasta(fasta)
    is_found = False
    for seqID, chromosomal_sequence in gen:
        print(seqID)
        if seqID == chromosome:
            is_found = True
            print(f"Chromosome {seqID} was located succesfully.")
            break
    if is_found == False:
        raise ValueError(f"Chromosome {chromosome} not found in the provided fasta.")
    return seqID, chromosomal_sequence

def schedule_sequences(sequences: list[dict], processes: int = 10) -> list[list]:
    jobs = [[] for _ in range(processes)]
    burden = [0 for _ in range(processes)]
    for seq in sequences:
        length = int(seq['length'])
        index = burden.index(min(burden))
        burden[index] += length
        jobs[index].append(seq)
    return jobs

def extract_controls(fasta: str,
                     chromosome: str,
                     sequence_path: str,
                     mode: str,
                     processes: int = 10) -> pd.DataFrame:
    seqID, chromosomal_sequence = search_chromosome(fasta, chromosome=chromosome)
    assert seqID == chromosome
    sequences = load_sequences(sequence_path=sequence_path)
    print(f"Total sequences detected: {len(sequences)}.")
    sequences_to_jobs = schedule_sequences(sequences, processes=processes)
    Path("progress").mkdir(exist_ok=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes,
                                                mp_context=multiprocessing.get_context('fork')) as executor:
        results = executor.map(_extract_controls, ((chromosomal_sequence, sequence, mode) for sequence in sequences_to_jobs))
        executor.shutdown(wait=True)
    tmp_files = [result for result in results]
    result_df = reduce(
                    lambda acc, val: pd.concat([acc, pd.read_table(val)]),
                    tmp_files,
                    pd.DataFrame()
                  ).sort_values(by=["chromosome", "start"], ascending=True)
    return result_df

def is_within_threshold(x: int, y: int, threshold: float = 0.05) -> bool:
    return ((1-threshold) * x <= y <= (1+threshold) * x) and ((1-threshold) * y <= x <= (1+threshold) * y)

def _extract_controls(args):
    chromosomal_sequence, sequences, mode = args
    Path(f"progress_{mode}").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s:%(levelname)s:%(message)s",
                        filename=f"progress_{mode}/progress_tracker_process_{os.getpid()}.txt")
    chromosome_length = len(chromosomal_sequence)
    tmp_file = tempfile.NamedTemporaryFile(delete=False,
                                           mode="w",
                                           prefix="controls")
    writer = csv.DictWriter(tmp_file,
                            delimiter="\t",
                            fieldnames=[
                                        "chromosome",
                                        "start",
                                        "end",
                                        "sequence",
                                        "controls_for",
                                        "length",
                                        "gc_content",
                                        "cg_counts",
                                        "gc_counts"
                                        ]
                            )
    writer.writeheader()
    # for sequence in tqdm(sequences, leave=True, position=0):
    total_sequences = len(sequences)
    completion = 0
    def _log():
        nonlocal completion
        while True:
            logging.info(f"Current progress from `{os.getpid()}`: {completion * 1e2 / total_sequences:.2f}")
            time.sleep(300)
    threading.Thread(target=_log, daemon=True).start()
    MAX_TRIES = 5_000_000
    threshold_flexibility = 0.05
    for i, sequence in enumerate(sequences):
        completion = i
        sequence_seq = sequence['sequence'].lower()
        sequence_c_content = sequence_seq.count('c') 
        sequence_g_content = sequence_seq.count('g')
        sequence_length = len(sequence_seq)
        if sequence_c_content == sequence_length or sequence_g_content == sequence_length:
            logging.info(f"Process id `{os.getpid()}` lost a g4.")
            continue
        sequence_gc_content = sequence_c_content + sequence_g_content
        sequence_cg_counts, sequence_gc_counts = count_occurrences(sequence_seq)
        for _ in range(samples_per_sequence):
            random_pos = random.randint(0, chromosome_length)
            step_generator = find_steps(random_pos,
                                        sequence_length,
                                        chromosome_length)
            for i, step in enumerate(step_generator):
                if i > MAX_TRIES:
                    logging.info(f"Process id `{os.getpid()}` lost a g4.")
                    break
                left, right = step
                control_candidate = chromosomal_sequence[left: right]
                control_candidate_gc_content = control_candidate.count('g') + control_candidate.count('c')
                control_candidate_cg_counts, control_candidate_gc_counts = count_occurrences(control_candidate)
                if is_within_threshold(control_candidate_gc_counts, sequence_gc_counts, threshold_flexibility) \
                        and is_within_threshold(control_candidate_cg_counts, sequence_cg_counts, threshold_flexibility) \
                        and is_within_threshold(control_candidate_gc_content, sequence_gc_content, threshold_flexibility):
                    writer.writerow({
                                "chromosome": chromosome,
                                "start": left,
                                "end": right,
                                "sequence": control_candidate,
                                "controls_for": sequence_seq,
                                "length": len(control_candidate),
                                "gc_content": control_candidate_gc_content,
                                "cg_counts": control_candidate_cg_counts,
                                "gc_counts": control_candidate_gc_counts,
                            })
                    break
    return tmp_file.name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""""")
    parser.print_help()
    parser.add_argument("--sequence_path", type=str,
            default="/storage/group/izg5139/default/external/quadrupia_database/sequence/nonBDNA/extraction/sequence_out/consensus_motifs/chm13v2.0.consensus.csv")
    parser.add_argument("--chromosome", type=str, default="chr1")
    parser.add_argument("--fasta", type=str)
    parser.add_argument("--mode", type=str, default="sequencehunter", choices=["g4hunter", "consensus"])
    parser.add_argument("--controls_path", type=str, default="controls_sequences")
    parser.add_argument("--bedtemp", type=str, default="/storage/home/nmc6088/scratch")
    parser.add_argument("--samples_per_sequence", type=int, default=2)
    parser.add_argument("--processes", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--max_tries", type=float, default=5_000_000)

    args = parser.parse_args()
    sequence_path = args.sequence_path
    chromosome = args.chromosome
    fasta = args.fasta
    mode = args.mode
    max_tries = args.max_tries
    processes = args.processes
    threshold = args.threshold
    controls_path = Path(args.controls_path).resolve()
    controls_path.mkdir(exist_ok=True)
    controls_path = controls_path.joinpath(f"controls.{mode}.{chromosome}.bed")
    # bedtemp = Path(args.bedtemp).resolve()
    # bedtemp.mkdir(exist_ok=True)

    samples_per_sequence = args.samples_per_sequence
    # if not bedtemp.is_dir():
    # raise ValueError(f"Invalid controls destination path {bedtemp}.")
    # pybedtools.helpers.set_tempdir(bedtemp)
    print(f"Generating sequence controls for chromosome '{chromosome}'.")
    print(f"Chosen error threshold: {threshold}.")
    controls_df = extract_controls(sequence_path=sequence_path,
                                   fasta=fasta,
                                   processes=processes,
                                   mode=mode,
                                   chromosome=chromosome)
    print("Control detection has been completed.")
    print(f"Saving sequence-free controls at {controls_path}...")
    controls_df = BedTool.from_dataframe(controls_df)
    human_sequence_df = pd.read_table(sequence_path, usecols=['seqID', 'start', 'end'])
    human_sequence = BedTool.from_dataframe(human_sequence_df)
    intersect_df = pd.read_table(
                                 controls_df.intersect(human_sequence, wao=True).fn,
                                 header=None,
                                 names=INTERSECT_FIELDS
                                )
    intersect_df = intersect_df[intersect_df['overlap'] == 0].drop(columns=['chromosome', 'mStart', 'mEnd', 'overlap'])
    intersect_df.to_csv(controls_path, mode="w", sep="\t", index=False)
    print("Controls have been succesfully saved on disk.")
    print("EXITING...")

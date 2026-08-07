# Landscape and mutational dynamics of G-quadruplexes across the complete human genome

Code and analysis notebooks accompanying the manuscript *"Landscape and mutational dynamics of G-quadruplexes across the complete human genome"*.

## About

G-quadruplexes (G4s) are alternative, non-B DNA structures that had long been difficult to study in repeat-rich regions of the genome due to the limitations of short-read sequencing. This repository contains the analysis code used to map and characterize potential G4-forming sequences across the gap-free T2T-CHM13 reference genome and across 88 haplotypes from the Human Pangenome Reference Consortium (HPRC), spanning diverse ancestries.

Using this pipeline, we show that G4s are strongly enriched in specific repetitive elements — including particular centromeric/pericentromeric satellite families and ribosomal DNA arrays — and experimentally validate the most prevalent predicted structures. Genome-wide, G4 loci tend to be hypomethylated relative to background and are disproportionately affected by insertions and deletions, indicating elevated genomic instability. We further find that G4s are consistently enriched at PRDM9 binding sites, a key determinant of meiotic recombination hotspots. Together, these results position G4s as dynamic, functionally relevant genomic elements with implications for genome evolution and disease.

## Repository structure

```
.
├── scripts/          # Core analysis pipeline (motif discovery, PWM/density modeling,
│                      pangenome motif extraction, methylation mapping, mutation
│                      enrichment, PRDM9/centromere analyses)
├── notebooks/         # Jupyter notebooks used to generate the paper's results and figures
├── data/               # Processed input/output data (methylation, pangenome, PRDM9
│                      hotspot densities, motif occurrences, model training artifacts)
├── figures/            # Genome-wide G4 density figures per chromosome
├── Paper_figures/      # Final, publication-ready figure files
├── model_training.tar.gz
└── requirements.txt
```

### Notebooks

| Notebook | Description |
|---|---|
| `Trinucleotide_Model.ipynb` | Trinucleotide-content background model for expected G4 density |
| `Methylation_G4_T2T.ipynb` | G4 methylation levels across the T2T genome |
| `Methylation_Rep_Time.ipynb` | G4 methylation vs. replication timing |
| `Mut_Analysis.ipynb`, `Mutation_densities_1kb.ipynb`, `Mutations-Fold-Enrichment-vs-Controls.ipynb` | Mutational instability (SNVs/indels) at G4 loci vs. controls |
| `Pangenomes-Haplotypes.ipynb` | G4 conservation/variation across 88 HPRC haplotypes |
| `PRMD9_hotspots_eG4.ipynb`, `PRMD9-Conservation_g4.ipynb` | Enrichment of G4s at PRDM9 binding sites/meiotic recombination hotspots |

## Figures

The publication-ready figure for the pangenome analysis is available in [`Paper_figures/pangenome_figure_1.pdf`](Paper_figures/pangenome_figure_1.pdf).

## Requirements

Dependencies are listed in [`requirements.txt`](requirements.txt) and can be installed with:

```bash
pip install -r requirements.txt
```

## License

This code is released under the [MIT License](LICENSE), an [OSI-approved](https://opensource.org/licenses/MIT) open source license.

## Contact

Nikol Chantzi (first author) — nicolechantzi@gmail.com

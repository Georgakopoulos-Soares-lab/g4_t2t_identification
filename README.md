# Identification of G4s across T2T Genomes

## Table of Contents

- [Haplotype Analysis](#haplotype-analysis)
- [Methylation Analysis](#methylation-analysis)
- [Mutation Analysis](#mutation-analysis)
- [PRMD9 Analysis](#prmd9-analysis)

## Abstract

G-quadruplexes (G4s) are alternative DNA structures with diverse biological roles, but their examination in highly repetitive parts of the human genome has been hindered by the lack of reliable sequencing technologies. Recent long-read based genome assemblies have enabled their characterization in previously inaccessible parts of the human genome. Here, we examine the topography and genomic instability of potential G4-forming sequences in the gap-less, reference human genome assembly and in 88 haplotypes of diverse ancestry. We report that G4s are highly enriched in specific repetitive regions, including in certain centromeric and pericentromeric repeat types, and in ribosomal DNA arrays, and experimentally validate the most prevalent G4s detected. G4s tend to have lower methylation than expected throughout the human genome and are genomically unstable, showing an excess of all mutation types, including substitutions, insertions and deletions and most prominently structural variants. Finally, we show that G4s are consistently enriched at PRDM9 binding sites, a protein involved in meiotic recombination. Together, our findings establish G4s as dynamic and functionally significant elements of the human genome and highlight new avenues for investigating their contributions to human disease and evolution.

## Haplotype Analysis

We analyzed 88 phased diploid haplotypes from the Human Pangenome Reference Consortium to study the distribution of G4s in T2T assemblies.
This involved comprehensive extraction and comparison of G4 motifs across diverse human haplotypes, enabling us to assess the variability, conservation, and population-specific patterns of G4s.
Custom scripts and pipelines were used to process large-scale haplotype data, identify unique and shared G4 motifs, and quantify their prevalence across different superpopulations. Results from this analysis provide insights into the evolutionary dynamics and functional significance of G4s in the context of human genetic diversity.

## Methylation Analysis

Methylation patterns surrounding G4s were modeled using Dirichlet distributions to capture probabilistic variation across sites.
We examined the replication timing and compared the methylation level of G4s versus the control group, focusing on both hypermethylated and hypomethylated regions.
This analysis leveraged high-resolution methylation datasets to determine whether G4s are associated with distinct epigenetic states, and to explore the relationship between G4 presence and local DNA methylation landscapes.
Our findings highlight the interplay between G4 structures and epigenetic regulation in the human genome.

## Mutation Analysis

We evaluated the mutational landscape around G4s, observing elevated rates of substitutions, indels, and structural variants.
In particular, we studied the following mutation types:

- Substitutions
- Small Insertions
- Small Deletions
- Structural Variants
- Multiple Nucleotide Polymorphisms

Using a combination of custom scripts and public variant datasets, we mapped mutations to G4-rich and control regions, quantified mutation rates, and assessed enrichment or depletion patterns.
This allowed us to investigate the potential role of G4s in promoting genomic instability and to identify mutation signatures that may be linked to G4 formation.

In-house python scripts were developed to validate and analyze the mutational landscape around G4s.

## Control Group

A matched control group of non-G4 genomic regions was selected to compare methylation, mutational, and enrichment patterns.
We used a randomized approach to identify sequences that share a similar distributional profile to the G4 group in regards to GC-content, CpG and GpC content, and length.
This careful matching ensures that observed differences between G4 and control regions are not confounded by basic sequence properties, allowing for robust comparative analyses.

- Control region scripts: [`scripts/controls_parallel_gc_content.py`](scripts/controls_parallel_gc_content.py)

## PRMD9 Analysis

We used long read sequencing data from Aleva et al. to analyze the distribution of G4s proximal to PRMD9 binding sites.
This analysis focused on the intersection between G4 motifs and PRMD9-associated recombination hotspots, providing insights into the potential mechanistic links between G4 formation and meiotic recombination processes.
Our results suggest a consistent enrichment of G4s at PRMD9 binding sites, supporting a functional relationship between these genomic features.

We used long read sequencing data from Aleva et. al to analyze the distribution of G4s proximal to PRMD9 binding sites.
This analysis focused on the intersection between G4 motifs and PRMD9-associated binding sites, providing insights into the potential mechanistic links between G4 formation and meiotic recombination processes.

## Contact

For any questions about the manuscript or the code please contact Nikol Chantzi or Dr. Ilias Georgakopoulos-Soares.:

```
Nikol Chantzi nmc6088@psu.edu nicolechantzi@gmail.com
Ilias Georgakopoulos-Soares izg5139@psu.edu
```

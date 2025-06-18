# Identification of G4s across T2T Genomes

## Table of Contents

- [Haplotype Analysis](#haplotype-analysis)
- [Methylation Analysis](#methylation-analysis)
- [Mutation Analysis](#mutation-analysis)
- [PRMD9 Analysis](#prmd9-analysis)

## Abstract

```
G-quadruplexes (G4s) are alternative DNA structures with diverse biological roles, but their examination in highly repetitive parts of the human genome has been hindered by the lack of reliable sequencing technologies. Recent long-read based genome assemblies have enabled their characterization in previously inaccessible parts of the human genome. Here, we examine the topography and genomic instability of potential G4-forming sequences in the gap-less, reference human genome assembly and in 88 haplotypes of diverse ancestry. We report that G4s are highly enriched in specific repetitive regions, including in certain centromeric and pericentromeric repeat types, and in ribosomal DNA arrays, and experimentally validate the most prevalent G4s detected. G4s tend to have lower methylation than expected throughout the human genome and are genomically unstable, showing an excess of all mutation types, including substitutions, insertions and deletions and most prominently structural variants. Finally, we show that G4s are consistently enriched at PRDM9 binding sites, a protein involved in meiotic recombination. Together, our findings establish G4s as dynamic and functionally significant elements of the human genome and highlight new avenues for investigating their contributions to human disease and evolution.
```

## Haplotype Analysis

We analyzed 88 phased diploid haplotypes from the Human Pangenome Reference Consortium to study the distribution of G4s in T2T assemblies.

## Methylation Analysis

Methylation patterns surrounding G4s were modeled using Dirichlet distributions to capture probabilistic variation across sites.

## Mutation Analysis

We evaluated the mutational landscape around G4s, observing elevated rates of substitutions, indels, and structural variants.

## Control Group

A matched control group of non-G4 genomic regions was selected to compare methylation, mutational, and enrichment patterns.

## PRMD9 Analysis

We used long read sequencing data from Aleva et. al to analyze the distribution of G4s proximal to 
PRMD9 binding sites.

## Contact

For any questions about the manuscript or the code please contact:

```
Nikol Chantzi nmc6088@psu.edu
Ilias Georgakopoulos-Soares izg5139@psu.edu
```
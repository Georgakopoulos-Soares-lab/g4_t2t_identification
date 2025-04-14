from tqdm import tqdm
import uuid
from collections import defaultdict
import gzip
import pandas as pd
from pathlib import Path

vcf_path = Path("/storage/group/izg5139/default/nicole/datasets/hprc-v1.1-mc-chm13.vcfbub.a100k.wave.vcf.gz")
def parse_VCF(vcf_path):
    with gzip.open(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("##"):
                continue

            header = line.strip().split("\t")[:9]
            break
        df_vcf = pd.read_table(f, header=None, usecols=range(9), names=header)
    return df_vcf

df_vcf = parse_VCF(vcf_path)
print(df_vcf.shape)

def classify_variant(allele: str, 
                     variant: str, 
                     smallins_thresh: int = 51, 
                     smalldel_thresh: int = 51) -> str:
    allele_N = len(allele)
    variant_N = len(variant)
    if allele_N == variant_N:
        return "snp" if allele_N == 1 else "mnp"
    if allele_N < variant_N:
        if allele_N == 1:
            return "smallins" if variant_N < smallins_thresh else "ins"
        else:
            return "complexins"
    if allele_N > variant_N:
        if variant_N == 1:
            return "smalldel" if allele_N < smalldel_thresh else "del"
        else:
            return "complexdel"
    raise ValueError()

def parse_attributes(attrs):
    attributes = defaultdict(dict)
    valid_attributes = {"AC", "AF", "AN", "LEN", "TYPE", "NS"}
    attrs = attrs.split(";")
    for aa in attrs:
        key, val = aa.split("=")
        if key == "NS" or key == "AN":
            attributes[key][0] = int(val)
            continue
            
        for i, v in enumerate(val.split(",")):   
            if key in valid_attributes:
                if key == "AC" or key == "LEN":
                    attributes[key][i] = int(v)
                elif key == "AF":
                    attributes[key][i] = float(v)
                else:
                    attributes[key][i] = v
    return attributes
    
def parse_VCF():
    vcf_enriched = []
    for _, row in tqdm(df_vcf.iterrows(), 
                       total=df_vcf.shape[0]):
        seqID = row['#CHROM']
        reference = row['REF']
        var = row['ALT']
        info = row['INFO']
        pos = row['POS'] - 1
        ref_len = len(reference)
        
        if isinstance(var, float):
            print(f"Skipping because variant is NAN! {row}.")
            continue

        var = var.split(",")
        attributes = parse_attributes(info)
        subvariants = len(var)
        unique_id = uuid.uuid4().hex[:8] 
        # adjust positions
        start = pos
        if ref_len == 1:
            end = pos + 1
        else:
            end = start + ref_len
            
        constant_attrs = {"AN", "NS"}
        valid_attributes = {"AC", "AF", "LEN", "TYPE"}    
        for i in range(subvariants):
            variant = var[i]
            specific_mutation = classify_variant(reference, variant)
            mutation = "ins" if specific_mutation.endswith("ins") \
                        else ("del" if specific_mutation.endswith("del") else specific_mutation)

            attrs = {key: attributes[key][i] for key in valid_attributes if key in attributes} | {key: attributes[key][0] for key in constant_attrs}
            vcf_enriched.append({
                        "seqID": seqID,
                        "start": start,
                        "end": end,
                        "mutation": specific_mutation,
                        "general_mut": mutation,
                        "subvariants": subvariants,
                        "mut_id": unique_id,
                        } \
                    | attrs | {
                        "reference": reference,
                        "variant": variant
                    }
                )
            
    vcf_enriched = pd.DataFrame(vcf_enriched)
    return vcf_enriched

vcf_enriched = parse_VCF()
vcf_enriched.to_csv("hprc-v1.1-mc-chm13.vcfbub.a100k.wave.exploded.vcf.gz", compression="gzip", sep="\t", mode="w", index=False)

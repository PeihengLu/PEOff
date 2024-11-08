import pandas as pd
import numpy as np
import genebe as gnb
import pyensembl
from crispAI.crispAI_score.annotate_pi import annotation_pipeline

# produce the dataset for training and testing
# first convert the hgvs variant to gene coordinate
def parse_hgvs(data: pd.DataFrame) -> pd.DataFrame:
    
    # locate the HGVS column
    hgvs = data.loc[:, 'HGVS']
    # convert to JSON serializable format
    hgvs = hgvs.to_list()

    # perform the conversion using genebe
    parsed_varients = gnb.parse_variants(hgvs)

    # split parsed varients into separate columns using '-'
    parsed_varients = [x.split('-') for x in parsed_varients]
    parsed_varients = [('chr'+ x[0], int(x[1])) for x in parsed_varients]

    # split the parsed varients into separate columns
    data[['chr', 'pos']] = pd.DataFrame(parsed_varients, index=data.index)

    return data

def convert_pridict_to_crispai(data: pd.DataFrame) -> pd.DataFrame:
    # add coordinates of the edit
    data = parse_hgvs(data)

    genes = data['Gene'].tolist()
    strand = []
    ensembl = pyensembl.EnsemblRelease(77)

    print(genes[:5])

    for gene in genes:
        try:
            gene = ensembl.genes_by_name(gene)
            # print("Gene:", gene.name)
            # print("Location:", gene.contig, gene.start, "-", gene.end)
            # print("Strand:", "Forward" if gene.strand == 1 else "Reverse")
            strand.append('Fw' if gene.strand == 1 else 'Rv')
        except:
            strand.append(None)

    target_strand = data['Target_Strand'].tolist()
    # remove the "'" from the strand
    target_strand = [s.replace("'", '') for s in target_strand]

    # add a strand column to the dataset
    # if target strand matches the gene strand, it is a sense edit
    data['strand'] = ['+' if s == t else '-' for s, t in zip(strand, target_strand)]

    # use the forward or reverse of the target strand and the Editing_Position column to identify the gRNA binding region
    # the 23 bp long regions are marked with start and end columns, inclusive
    data['start'] = data['pos'] - data['Editing_Position'] - data['PBSlength'] + 1 if strand == 'Fw' else data['pos'] + data['Editing_Position'] + data['PBSlength'] - 22
    data['end'] = data['start'] + 22

    data['target_sequence'] = data['wide_initial_target'].str.slice(10, 33)
    data['sgRNA_sequence'] = data['target_sequence']

    data = data[['chr', 'start', 'end', 'strand', 'target_sequence', 'sgRNA_sequence']]

    data = annotation_pipeline(data)

    return data
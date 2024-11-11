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
    genebes = gnb.parse_variants(hgvs)

    # split parsed varients into separate columns using '-'
    parsed_varients = [x.split('-') for x in genebes]
    parsed_varients = [('chr'+ x[0], int(x[1])) for x in parsed_varients]

    # split the parsed varients into separate columns
    data[['chr', 'pos']] = pd.DataFrame(parsed_varients, index=data.index)

    data['pos'] = data['pos'].astype(int)
    data['genebe'] = genebes
    # TODO: Check if this is necessary
    # data['pos'] = data['pos'] + 1 if 'del' in data['HGVS'].values else data['pos']
    # data['pos'] = data['pos'] + 1 if 'dup' in data['HGVS'].values else data['pos']

    return data

def convert_pridict_to_crispai(data: pd.DataFrame) -> pd.DataFrame:
    # add coordinates of the edit
    data = parse_hgvs(data)

    genes = data['Gene'].tolist()
    strand = []

    ensemble = pyensembl.EnsemblRelease(111)

    for gene in genes:
        try:
            gene_data = ensemble.genes_by_name(gene)[0]
            # print("Gene:", gene.name)
            # print("Location:", gene.contig, gene.start, "-", gene.end)
            strand.append(gene_data.strand)
        except:
            print(f"{gene} not found")
            strand.append(None)

    target_strand = data['Target_Strand'].tolist()
    # remove the "'" from the strand
    target_strand = [s.replace("'", '') for s in target_strand]

    def opposite_strand(strand: str) -> str:
        return '+' if strand == '-' else '-'

    # add a strand column to the dataset
    # if target strand matches the gene strand, it is a sense edit
    data['og_strand'] = strand
    data['direction'] = target_strand
    data.loc[(data['direction'] == 'Fw'), 'strand'] = '+'
    data.loc[~(data['direction'] == 'Fw'), 'strand'] = '-'
    
    # use the forward or reverse of the target strand and the Editing_Position column to identify the gRNA binding region
    # the 23 bp long regions are marked with start and end columns, inclusive
    data.loc[(data['strand'] == '+'), 'start'] = data['pos'] - (data['RTlength'] - data['RToverhanglength']) - data['PBSlength']
    data.loc[(data['strand'] == '-'), 'start'] = data['pos'] + (data['RTlength'] - data['RToverhanglength']) + data['PBSlength'] - 22
    data.loc[(data['strand'] == '-') & (data['Correction_Type'] == 'Insertion'), 'start'] = data['start'] + 1
    data.loc[(data['strand'] == '-') & (data['Correction_Type'] == 'Deletion'), 'start'] = data['start'] + 1
    data.loc[(data['strand'] == '+') & (data['Correction_Type'] == 'Insertion'), 'start'] = data['start'] - 1 + data['Correction_Length']
    data.loc[(data['strand'] == '+') & (data['Correction_Type'] == 'Deletion'), 'start'] = data['start'] - 1 + data['Correction_Length']

    # save the data to the original file

    # data.loc[(data['strand'] == '+') & (data['Correction_Type'] == 'Deletion'), 'start'] 

    data['start'] = data['start'].astype(int)
    data['end'] = data['start'] + 22

    data['target_sequence'] = data['wide_mutated_target'].str.slice(10, 33)
    data['sgRNA_sequence'] = data['target_sequence']

    data = data[['chr', 'start', 'end', 'strand', 'target_sequence', 'sgRNA_sequence', 'genebe']]

    data = annotation_pipeline(data)

    return data
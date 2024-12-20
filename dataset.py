import os

import pandas as pd
import numpy as np
import genebe as gnb
import pyensembl
from crispAI.crispAI_score.annotate_pi import annotation_pipeline

# produce the dataset for training and testing
# first convert the hgvs variant to gene coordinate
def parse_hgvs(data: pd.DataFrame, genebe_annotations: pd.DataFrame) -> pd.DataFrame:
    # join the two dataframes on the HGVS column, delete if not existent in genebe_annotations
    data = data.merge(genebe_annotations, on='HGVS', how='inner')

    # find the genebe annotations where the second element is ''
    # print("Genebe annotations with empty second element")
    # print(data[data['genebe'].str.split('-').str[1] == ''])
    data = data[data['genebe'].str.split('-').str[1] != '']

    # split parsed varients into separate columns using '-'
    parsed_varients = [x.split('-') for x in data['genebe'].tolist()]
    parsed_varients = [('chr'+ x[0], int(x[1])) for x in parsed_varients]

    # split the parsed varients into separate columns
    data[['chr', 'pos']] = pd.DataFrame(parsed_varients, index=data.index)

    data['pos'] = data['pos'].astype(int)

    return data

def convert_pridict_to_crispai(data: pd.DataFrame, model:str = 'base') -> pd.DataFrame:
    # get the genebe annotations from 'data/genebe_annotations.csv'
    if os.path.exists('data/genebe_annotations.csv'):
        genebe_annotations = pd.read_csv('data/genebe_annotations.csv')
    else:
        genebe_annotations = pd.DataFrame()
        genebe_annotations['HGVS'] = data['HGVS']
        # remove duplicates
        genebe_annotations = genebe_annotations.drop_duplicates()
        genebe_annotations = genebe_annotations.reset_index(drop=True)
        genebe_annotations['genebe'] = gnb.parse_variants(genebe_annotations['HGVS'].to_list())
        # save the genebe annotations to 'data/genebe_annotations.csv'
        genebe_annotations.to_csv('data/genebe_annotations.csv', index=False)

    # look for nan values in the genebe column
    if genebe_annotations['genebe'].isnull().sum() > 0:
        print("Some genebe annotations are missing")
        genebe_annotations = genebe_annotations.dropna(subset=['genebe'])

    # add coordinates of the edit
    data = parse_hgvs(data, genebe_annotations)

    target_strand = data['Target_Strand'].tolist()
    # remove the "'" from the strand
    target_strand = [s.replace("'", '') for s in target_strand]

    # add a strand column to the dataset
    # if target strand matches the gene strand, it is a sense edit
    data['direction'] = target_strand
    data.loc[(data['direction'] == 'Fw'), 'strand'] = '+'
    data.loc[~(data['direction'] == 'Fw'), 'strand'] = '-'

    seq_len = 60 if model != 'base' else 23
    
    # use the forward or reverse of the target strand and the Editing_Position column to identify the gRNA binding region
    # the 23 bp long regions are marked with start and end columns, inclusive
    data.loc[(data['strand'] == '+'), 'start'] = data['pos'] - (data['RTlength'] - data['RToverhanglength']) - data['PBSlength'] + 1 
    data.loc[(data['strand'] == '-'), 'start'] = data['pos'] + (data['RTlength'] - data['RToverhanglength']) + data['PBSlength'] - seq_len + 1 
    data.loc[(data['strand'] == '-') & (data['Correction_Type'] == 'Insertion'), 'start'] = data['start'] 
    data.loc[(data['strand'] == '-') & (data['Correction_Type'] == 'Deletion'), 'start'] = data['start']
    data.loc[(data['strand'] == '+') & (data['Correction_Type'] == 'Insertion'), 'start'] = data['start'] + data['Correction_Length'] - 1
    data.loc[(data['strand'] == '+') & (data['Correction_Type'] == 'Deletion'), 'start'] = data['start'] + data['Correction_Length'] - 1

    data['start'] = data['start'].astype(int)
    data['end'] = data['start'] + seq_len - 1

    data['target_sequence'] = data['wide_mutated_target'].str.slice(10, 33)
    data['sgRNA_sequence'] = data['target_sequence']

    data = data[['chr', 'start', 'end', 'strand', 'target_sequence', 'sgRNA_sequence', 'genebe', 'PE2df_percentageedited', 'uniqueindex']]

    # rename pe2df_percentageedited to efficiency
    data = data.rename(columns={'PE2df_percentageedited': 'efficiency'})

    data = annotation_pipeline(data, model=model)

    return data
import pandas as pd
import numpy as np
import genebe as gnb

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
    data[['chrom', 'pos']] = pd.DataFrame(parsed_varients, index=data.index)

    return data
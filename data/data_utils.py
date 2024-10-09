# convert PRIDICT dataset into crispAI format
import pandas as pd
import os

def convert_pridict_diverse_to_crispAI_base() -> None:
    """convert PRIDICT dataset into crispAI format (exact format)
    """
    fold = 'testset_fold'
    
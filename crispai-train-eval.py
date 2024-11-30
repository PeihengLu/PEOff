from dataclasses import dataclass
from typing import Dict, Tuple
from os.path import join as pjoin, exists
import argparse

import skorch
from skorch import NeuralNetRegressor
from skorch.callbacks import LRScheduler, EarlyStopping, Checkpoint
import torch
import pandas as pd
import scipy.stats as stats
import numpy as np

from crispAI.crispAI_score.model import CrispAI_pi, ModelConfig, CrispAI_pe
from crispAI.crispAI_score.loss_functions import MyZeroInflatedNegativeBinomialLoss

import sys
sys.path.append('crispAI/crispAI_score')

# acquire the config
checkpoint = torch.load('crispAI/crispAI_score/model_checkpoint/epoch:19-best_valid_loss:0.270.pt', map_location=torch.device('cuda'))
model_config = checkpoint['config']
# model_config = ModelConfig()

@dataclass
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 0.01
    epochs: int = 100
    
    # lr scheduler
    use_scheduler: bool = True
    scheduler: LRScheduler = LRScheduler(policy='StepLR', monitor='valid_loss', step_size=5, gamma=0.1)
    
    # callbacks
    use_early_stopping: bool = True
    patience: int = 5
    use_checkpoint: bool = True
    f_params: str = 'model_params.pt'
    f_history: str = None
    f_criteria: str = None
    f_best: str = 'model_best.pt'
    
    # device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # custom criterion using zero-inflated negative binomial
    criterion: torch.nn.Module = MyZeroInflatedNegativeBinomialLoss
    
    # adam optimizer
    optimizer: torch.optim.Optimizer = torch.optim.Adam
    
def train_crispAI(training_config: TrainingConfig, X_trains: Dict[int, Dict[str, torch.Tensor]], y_trains: Dict[int, torch.Tensor], X_tests: Dict[int, Dict[str, torch.Tensor]], y_tests: Dict[int, torch.Tensor]) -> None:
    """train crispAI model on a set of cross validation splits

    Args:
        training_config (TrainingConfig): training configuration
        X_trains (Dict[int, Dict[str, torch.Tensor]]): training data for each split, where the key is the split index, and the value is a dictionary containing the input names and tensors for the forward function
        y_trains (Dict[int, torch.Tensor]): training labels for each split, where the key is the split index
        X_tests (Dict[int, Dict[str, torch.Tensor]]): test data for each split, where the key is the split index, and the value is a dictionary containing the input names and tensors for the forward function
        y_tests (Dict[int, torch.Tensor]): test labels for each split, where the key is the split index
    """    
    # cross validation with each fold
    for fold in X_trains.keys():
        print(f"Training fold {fold}")
        net = NeuralNetRegressor(
            CrispAI_pi(model_config),
            max_epochs=training_config.epochs,
            lr=training_config.learning_rate,
            batch_size=training_config.batch_size,
            device='cuda',
            train_split=skorch.dataset.ValidSplit(cv=5),
            callbacks=[
                # training_config.scheduler if training_config.use_scheduler else None,
                EarlyStopping(patience=training_config.patience, lower_is_better=True),
                Checkpoint(monitor='valid_loss_best', 
                           f_params=training_config.f_params + f'fold_{fold}.pt' if training_config.f_params is not None else None
                           , f_optimizer=None, f_history=training_config.f_history+ f'fold_{fold}.pt' if training_config.f_history is not None else None,
                           f_criterion=training_config.f_criteria+ f'fold_{fold}.pt' if training_config.f_criteria is not None else None, f_optimizer_state=None, f_criterion_state=None, f_best=training_config.f_best + f'fold_{fold}.pt' if training_config.f_best is not None else None)
            ],
            criterion=training_config.criterion,
            optimizer=training_config.optimizer,
        )
        net.initialize()
        # initialize optimizer
        
        # train the model
        net.fit(X_trains[fold], y_trains[fold])
        

def preprocess_data(data: pd.DataFrame, mode: str = 'train', model='base') -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, torch.Tensor], Dict[int, Dict[str, torch.Tensor]], Dict[int, torch.Tensor]]:
    """preprocess data for crispAI model

    Args:
        data (pd.DataFrame): data with all required columns
        
    Returns:
        Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, torch.Tensor]: training data and labels
    """    
    # TODO: split the data into folds based on uniqueindex field
    # TODO: may need to group edits on same target loci together in the future
    data['fold'] = data['uniqueindex'] % 5
    # sequence length
    seq_len = 23 if model == 'base' else 60

    # dictionaries to store the training data and labels for each fold
    X_trains = {}
    y_trains = {}
    X_tests = {}
    y_tests = {}

    # literal parsing of the list strings to lists of floats, Nucleotide BDM score, GC content, NuPoP occupancy score, and NuPoP affinity scores
    data['nucleotide BDM'] = data['nucleotide BDM'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])
    data['GC flank73'] = data['GC flank73'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])
    data['NuPoP occupancy'] = data['NuPoP occupancy'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])
    data['NuPoP affinity'] = data['NuPoP affinity'].apply(lambda x: [float(i) for i in x.strip('[]').split(',')])

    vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, '-': 5}

    print('Converted list strings to lists of floats')

    for fold in data['fold'].unique():
        print(f"Processing fold {fold}")
        train_data = data[data['fold'] != fold].reset_index(drop=True)
        test_data = data[data['fold'] == fold].reset_index(drop=True)
        # one hot encode the target and sgRNA sequence by taking the element wise OR between the target and sgRNA sequence
        # the final two dimensions indicating mismatch direction is unused here (because there are only perfect matches), so add two dummy rows of zeros
        X_trains[fold] = {
            'X_nucl': np.zeros((train_data.shape[0], seq_len, 6)),
            'X_pi': np.zeros((train_data.shape[0], seq_len, 4))
        }

        if mode == 'train':
            # element wise OR between target and protospacer, in this case, is just one hot encoding the sequence
            for i, row in train_data.iterrows():
                sequence = row['context sequence flank_73'][0: seq_len]
                sequence = sequence.upper()

                for j, s in enumerate(sequence):
                    X_trains[fold]['X_nucl'][i, j, vocab[s]] = 1
                # the last two columns is used for annotating the pbs and rt_mut location
                if model == 'annot':
                    pbs_loc = eval(row['pbs_loc'])
                    rt_mut_loc = eval(row['rt_mut_loc'])
                    # subtract 10 from the pbs_loc and rt_mut_loc to get the correct index
                    pbs_loc = [x - 10 for x in pbs_loc]
                    rt_mut_loc = [x - 10 for x in rt_mut_loc]

                    X_trains[fold]['X_nucl'][i, pbs_loc[0]: pbs_loc[1], -2] = 1
                    X_trains[fold]['X_nucl'][i, rt_mut_loc[0]: rt_mut_loc[1], -1] = 1
            print('One hot encoded training target and protospacer sequence')

            # load the Nucleotide BDM score, GC content, NuPoP occupancy score, and NuPoP affinity scores
            for i, row in train_data.iterrows():
                X_trains[fold]['X_pi'][i, :, 0] = row['GC flank73']
                X_trains[fold]['X_pi'][i, :, 1] = row['nucleotide BDM']
                X_trains[fold]['X_pi'][i, :, 2] = row['NuPoP occupancy']
                X_trains[fold]['X_pi'][i, :, 3] = row['NuPoP affinity']

            y_trains[fold] = torch.tensor(train_data['efficiency'].values, dtype=torch.float32)

            print('Loaded training Nucleotide BDM score, GC content, NuPoP occupancy score, and NuPoP affinity scores')

        # do the same for the test data
        X_tests[fold] = {
            'X_nucl': np.zeros((test_data.shape[0], seq_len, 6)),
            'X_pi': np.zeros((test_data.shape[0], seq_len, 4))
        }

        for i, row in test_data.iterrows():
            sequence = row['context sequence flank_73'][0: seq_len]
            # sequence to upper case
            sequence = sequence.upper()
            for j, s in enumerate(sequence):
                X_tests[fold]['X_nucl'][i, j, vocab[s]] = 1
            # the last two columns is used for annotating the pbs and rt_mut location
            if model == 'annot':
                pbs_loc = eval(row['pbs_loc'])
                rt_mut_loc = eval(row['rt_mut_loc'])
                # subtract 10 from the pbs_loc and rt_mut_loc to get the correct index
                pbs_loc = [x - 10 for x in pbs_loc]
                rt_mut_loc = [x - 10 for x in rt_mut_loc]
                
                X_tests[fold]['X_nucl'][i, pbs_loc[0]: pbs_loc[1], -2] = 1
                X_tests[fold]['X_nucl'][i, rt_mut_loc[0]: rt_mut_loc[1], -1] = 1
                
        print('One hot encoded test target and protospacer sequence')

        for i, row in test_data.iterrows():
            X_tests[fold]['X_pi'][i, :, 0] = row['GC flank73']
            X_tests[fold]['X_pi'][i, :, 1] = row['nucleotide BDM']
            X_tests[fold]['X_pi'][i, :, 2] = row['NuPoP occupancy']
            X_tests[fold]['X_pi'][i, :, 3] = row['NuPoP affinity']

        print('Loaded test Nucleotide BDM score, GC content, NuPoP occupancy score, and NuPoP affinity scores')

        y_tests[fold] = torch.tensor(test_data['efficiency'].values, dtype=torch.float32)

    # all data should be of type float32
    for fold in X_trains.keys():
        for key in X_trains[fold].keys():
            X_trains[fold][key] = torch.tensor(X_trains[fold][key], dtype=torch.float32)
        for key in X_tests[fold].keys():
            X_tests[fold][key] = torch.tensor(X_tests[fold][key], dtype=torch.float32)

    # check for nan values in any of the tensors
    for fold in X_trains.keys():
        for key in X_trains[fold].keys():
            if torch.isnan(X_trains[fold][key]).any():
                print(f"Found nan values in training data for fold {fold}")
        for key in X_tests[fold].keys():
            if torch.isnan(X_tests[fold][key]).any():
                print(f"Found nan values in test data for fold {fold}")

    # return the training data and labels
    return X_trains, y_trains, X_tests, y_tests


@dataclass
class TesingConfig:
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 100
    
    # lr scheduler
    use_scheduler: bool = True
    scheduler: LRScheduler = LRScheduler(policy='StepLR', monitor='valid_loss', step_size=5, gamma=0.1)
    
    # callbacks
    use_early_stopping: bool = True
    patience: int = 10
    use_checkpoint: bool = True
    f_params: str = 'model_params.pt'
    f_history: str = None
    f_criteria: str = None
    f_best: str = 'model_best.pt'
    
    # device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # custom criterion using zero-inflated negative binomial
    criterion: torch.nn.Module = MyZeroInflatedNegativeBinomialLoss
    
    # adam optimizer
    optimizer: torch.optim.Optimizer = torch.optim.Adam
    
def evaluate_crisp_ai(testing_config: TesingConfig, X_tests: Dict[int, Dict[str, torch.Tensor]], y_tests: Dict[int, torch.Tensor], model:str = 'base') -> None:
    """train crispAI model on a set of cross validation splits

    Args:
        training_config (TrainingConfig): training configuration
        X_trains (Dict[int, Dict[str, torch.Tensor]]): training data for each split, where the key is the split index, and the value is a dictionary containing the input names and tensors for the forward function
        y_trains (Dict[int, torch.Tensor]): training labels for each split, where the key is the split index
        X_tests (Dict[int, Dict[str, torch.Tensor]]): test data for each split, where the key is the split index, and the value is a dictionary containing the input names and tensors for the forward function
        y_tests (Dict[int, torch.Tensor]): test labels for each split, where the key is the split index
    """    
    pearson_correlations = {}
    spearman_correlations = {}
    # cross validation with each fold
    for fold in X_tests.keys():
        print(f"Testing fold {fold}")
        net = NeuralNetRegressor(
            CrispAI_pi(model_config),
            max_epochs=testing_config.epochs,
            lr=testing_config.learning_rate,
            batch_size=testing_config.batch_size,
            device='cuda',
            criterion=testing_config.criterion,
            optimizer=testing_config.optimizer,
        )
        net.initialize()
        # load the trained model
        net.load_params(f_params=testing_config.f_params + f'fold_{fold}.pt')

        # output the test set predictions performance using pearson and spearman correlation
        # sample 1000 times to get the measured efficiency and confidence interval
        sampled_efficiencies = net.module.draw_samples(X_tests[fold], 2000)
        # take the mean across the first dimension
        mean_efficiencies = np.mean(sampled_efficiencies, axis=0)

        # convert target efficiencies to numpy
        y_tests[fold] = y_tests[fold].cpu().numpy()

        # calculate the pearson and spearman correlation with the target
        print(f"Pearson correlation: {stats.pearsonr(mean_efficiencies, y_tests[fold])[0]}")
        print(f"Spearman correlation: {stats.spearmanr(mean_efficiencies, y_tests[fold])[0]}")
        
        pearson_correlations[fold] = stats.pearsonr(mean_efficiencies, y_tests[fold])[0]
        spearman_correlations[fold] = stats.spearmanr(mean_efficiencies, y_tests[fold])[0]
    
    
    # save the pearson and spearman correlations to csv file
    pearson_performance = pd.read_csv('data/pridict_90k_pearson.csv') if exists('data/pridict_90k_pearson.csv') else pd.DataFrame()
    spearman_performance = pd.read_csv('data/pridict_90k_spearman.csv') if exists('data/pridict_90k_spearman.csv') else pd.DataFrame()
    
    
    # the performance files uses fold as index
    for fold in pearson_correlations.keys():
        pearson_performance.at[fold, f'crispAI-{model}'] = pearson_correlations[fold]
        spearman_performance
        spearman_performance.at[fold, f'crispAI-{model}'] = spearman_correlations[fold] 
        
    pearson_performance.to_csv('data/pridict_90k_pearson.csv', index=False)
    spearman_performance.to_csv('data/pridict_90k_spearman.csv', index=False)


if __name__ == '__main__':
    # parse cmd arguments
    parser = argparse.ArgumentParser(description='Train crispAI model on PRIDICT dataset')
    parser.add_argument('--data', type=str, default='data/crispai-90k-filtered.csv', help='path to the PRIDICT dataset')
    parser.add_argument('--output', type=str, default='crispAI/trained_models', help='path to save the trained model')
    parser.add_argument('--model', type=str, default='base', help='model configuration to use', choices=['base', 'long', 'annot'])
    parser.add_argument('--distribution', type=str, default='negative_binomial', help='distribution to use for the loss function', choices=['negative_binomial'])
    parser.add_argument('--mode', type=str, default='train', help='mode to run the script in', choices=['train', 'eval'])
    # test flag
    parser.add_argument('--test', action='store_true', help='flag to test the script')
    args = parser.parse_args()
    
    # load the data
    if args.model == 'base':
        data = pd.read_csv('data/crispai-90k.csv')
    elif args.model == 'long':
        data = pd.read_csv('data/crispai-90k-long.csv')
    elif args.model == 'annot':
        data = pd.read_csv('data/crispai-90k-long.csv')
        # load the pridict-90k data and keep the columns needed for the model
        annot_data = pd.read_csv('data/pridict-90k.csv')
        annot_data = annot_data[['uniqueindex', 'PBSlocation', 'RT_mutated_location']]
        # rename the PBSlocation and RT_mutated_location columns to pbs_loc and rt_mut_loc
        annot_data.rename(columns={'PBSlocation': 'pbs_loc', 'RT_mutated_location': 'rt_mut_loc'}, inplace=True)
        # join the data with the annotation data
        data = data.merge(annot_data, on='uniqueindex')

    # sample 1000 rows for testing
    if args.test:
        data = data.sample(1000)

    # preprocess the data
    X_trains, y_trains, X_tests, y_tests = preprocess_data(data, mode=args.mode, model=args.model)

    if args.mode == 'train':
        training_config = TrainingConfig()
        
        if args.model == 'base':
            training_config.f_params = pjoin('crispAI', 'trained_models', 'pridict_params_')
            training_config.f_best = pjoin('crispAI', 'trained_models', 'pridict_best_')
        elif args.model == 'long':
            training_config.f_params = pjoin('crispAI', 'trained_models', 'pridict_params_long_')
            training_config.f_best = pjoin('crispAI', 'trained_models', 'pridict_best_long_')
            
            model_config.seq_len = 60
        elif args.model == 'annot':
            training_config.f_params = pjoin('crispAI', 'trained_models', 'pridict_params_annot_')
            training_config.f_best = pjoin('crispAI', 'trained_models', 'pridict_best_annot_')
            
            model_config.seq_len = 60
            
        # distribution setting
        if args.distribution == 'negative_binomial':
            training_config.criterion = MyZeroInflatedNegativeBinomialLoss
        else:
            # TODO: update this with other distributions after a good one is decided
            training_config.criterion = torch.nn.MSELoss
        # train the model
        train_crispAI(training_config=training_config, X_trains=X_trains, y_trains=y_trains, X_tests=X_tests, y_tests=y_tests)
    if args.mode == 'eval':
        # TODO: modify the config according to the cmd arguments
        # attach all data to cuda
        for fold in X_tests.keys():
            for key in X_tests[fold].keys():
                X_tests[fold][key] = X_tests[fold][key].to('cuda')
            y_tests[fold] = y_tests[fold].to('cuda')

        testing_config = TesingConfig()
        model_config.conv_dropout = 0
        model_config.lstm_dropout = 0
        if args.model == 'base':
            testing_config.f_params = pjoin('crispAI', 'trained_models', 'pridict_params_')
            testing_config.f_best = pjoin('crispAI', 'trained_models', 'pridict_best_')
        elif args.model == 'long':
            testing_config.f_params = pjoin('crispAI', 'trained_models', 'pridict_params_long_')
            testing_config.f_best = pjoin('crispAI', 'trained_models', 'pridict_best_long_')
            
            model_config.seq_len = 60
        elif args.model == 'annot':
            testing_config.f_params = pjoin('crispAI', 'trained_models', 'pridict_params_annot_')
            testing_config.f_best = pjoin('crispAI', 'trained_models', 'pridict_best_annot_')
            
            model_config.seq_len = 60
        # train the model
        evaluate_crisp_ai(testing_config, X_tests, y_tests, model=args.model)   
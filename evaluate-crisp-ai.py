from dataclasses import dataclass
from typing import Dict, Tuple
from os.path import join as pjoin, exists

import skorch
from skorch import NeuralNetRegressor
from skorch.callbacks import LRScheduler, EarlyStopping, Checkpoint
import torch
import pandas as pd
import scipy.stats as stats
import numpy as np

from crispAI.crispAI_score.model import CrispAI_pi, ModelConfig
from crispAI.crispAI_score.loss_functions import ZeroInflatedNegativeBinomialLoss

# set dropout to 0 for evaluation
import sys
sys.path.append('crispAI/crispAI_score')

# acquire the config from the saved model
checkpoint = torch.load('crispAI/crispAI_score/model_checkpoint/epoch:19-best_valid_loss:0.270.pt', map_location=torch.device('cuda'))
model_config = checkpoint['config']

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
    criterion: torch.nn.Module = ZeroInflatedNegativeBinomialLoss
    
    # adam optimizer
    optimizer: torch.optim.Optimizer = torch.optim.Adam
    
def evaluate_crisp_ai(testing_config: TesingConfig, X_tests: Dict[int, Dict[str, torch.Tensor]], y_tests: Dict[int, torch.Tensor]) -> None:
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
            train_split=skorch.dataset.ValidSplit(cv=5),
            callbacks=[
                EarlyStopping(patience=testing_config.patience, lower_is_better=True),
                Checkpoint(monitor='valid_loss_best', 
                           f_params=testing_config.f_params + f'fold_{fold}.pt' if testing_config.f_params is not None else None
                           , f_optimizer=None, f_history=testing_config.f_history+ f'fold_{fold}.pt' if testing_config.f_history is not None else None,
                           f_criterion=testing_config.f_criteria+ f'fold_{fold}.pt' if testing_config.f_criteria is not None else None, f_optimizer_state=None, f_criterion_state=None, f_best=testing_config.f_best + f'fold_{fold}.pt' if testing_config.f_best is not None else None)
            ],
            criterion=testing_config.criterion,
            optimizer=testing_config.optimizer,
        )
        net.initialize()
        # load the trained model
        net.load_params(f_params=testing_config.f_params + f'fold_{fold}.pt')

        # output the test set predictions performance using pearson and spearman correlation
        # sample 100 times to get the measured efficiency and confidence interval
        sampled_efficiencies = net.module.draw_samples(X_tests[fold], 100)


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
        pearson_performance.at[fold, 'crispAI'] = pearson_correlations[fold]
        spearman_performance
        spearman_performance.at[fold, 'crispAI'] = spearman_correlations[fold] 
        
    pearson_performance.to_csv('data/pridict_90k_pearson.csv', index=False)
    spearman_performance.to_csv('data/pridict_90k_spearman.csv', index=False)
            

def preprocess_data(data: pd.DataFrame) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, torch.Tensor]]:
    """preprocess data for crispAI model

    Args:
        data (pd.DataFrame): data with all required columns
        
    Returns:
        Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, torch.Tensor]: training data and labels
    """    
    # TODO: split the data into folds based on uniqueindex field
    # TODO: may need to group edits on same target loci together in the future
    # remove the entries with 0 efficiency
    data['fold'] = data['uniqueindex'] % 5

    # dictionaries to store the training data and labels for each fold
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
        test_data = data[data['fold'] == fold].reset_index(drop=True)

        X_tests[fold] = {
            'X_nucl': np.zeros((test_data.shape[0], 23, 6)),
            'X_pi': np.zeros((test_data.shape[0], 23, 4))
        }

        for i, row in test_data.iterrows():
            sequence = row['target_sequence']

            for j, s in enumerate(sequence):
                X_tests[fold]['X_nucl'][i, j, vocab[s]] = 1

        print('One hot encoded test target and protospacer sequence')

        for i, row in test_data.iterrows():
            X_tests[fold]['X_pi'][i, :, 0] = row['GC flank73']
            X_tests[fold]['X_pi'][i, :, 1] = row['nucleotide BDM']
            X_tests[fold]['X_pi'][i, :, 2] = row['NuPoP occupancy']
            X_tests[fold]['X_pi'][i, :, 3] = row['NuPoP affinity']

        print('Loaded test Nucleotide BDM score, GC content, NuPoP occupancy score, and NuPoP affinity scores')

        y_tests[fold] = torch.tensor(test_data['efficiency'].values, dtype=torch.float32)

    # all data should be of type float32
    for fold in X_tests.keys():
        for key in X_tests[fold].keys():
            X_tests[fold][key] = torch.tensor(X_tests[fold][key], dtype=torch.float32)


    # return the training data and labels
    return X_tests, y_tests


if __name__ == '__main__':
    # load the data
    data = pd.read_csv('data/crispai-90k-filtered.csv')

    # save the uniqueid for later
    uniqueid = {}

    for fold in range(5):
        uniqueid[fold] = data[data['uniqueindex'] % 5 == fold]['uniqueindex']

    # preprocess the data
    X_tests, y_tests = preprocess_data(data)

    # attach all data to cuda
    for fold in X_tests.keys():
        for key in X_tests[fold].keys():
            X_tests[fold][key] = X_tests[fold][key].to('cuda')
        y_tests[fold] = y_tests[fold].to('cuda')

    testing_config = TesingConfig()
    testing_config.f_params = pjoin('crispAI', 'trained_models', 'pridict_params_')
    testing_config.f_best = pjoin('crispAI', 'trained_models', 'pridict_best_')
    # train the model
    evaluate_crisp_ai(testing_config, X_tests, y_tests)
from dataclasses import dataclass
from typing import Dict, Tuple

from skorch import NeuralNetRegressor
from skorch.callbacks import LRScheduler
import torch
import pandas as pd

from crispAI.crispAI_score.model import CrispAI_pi, ModelConfig


config = ModelConfig()

class ZeroInflatedNegativeBinomial(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super(ZeroInflatedNegativeBinomial, self).__init__()
        self.eps = eps

    def negative_binomial(self, y, mu, theta):
        return (torch.lgamma(y + theta) / torch.lgamma(theta)) * (theta / (theta + mu)) ** theta * (mu / (theta + mu)) ** y 
        
    def forward(self, y_pred, y_true):
        y_pred = torch.clamp(y_pred, min=self.eps)
        
        pi = y_pred[:, 0]
        mu = y_pred[:, 1]
        theta = y_pred[:, 2]

        y = y_true

        # mixture of zero and negative binomial
        zero = torch.log(pi + self.eps) + torch.log(1 + torch.exp(-mu))
        non_zero = torch.log(1 - pi + self.eps) + torch.log(self.negative_binomial(y, mu, theta) + self.eps)

        return -torch.mean(zero * (y == 0).float() + non_zero * (y > 0).float())

@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 10
    
    # lr scheduler
    use_scheduler: bool = True
    scheduler: LRScheduler = LRScheduler(policy='StepLR', monitor='valid_loss', step_size=5, gamma=0.1)
    
    # callbacks
    use_early_stopping: bool = True
    patience: int = 5
    use_checkpoint: bool = True
    f_params: str = 'model_params.pt'
    f_history: str = 'model_history.json'
    f_criteria: str = 'model_criteria.json'
    f_best: str = 'model_best.pt'
    
    # device
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # custom criterion using zero-inflated negative binomial
    criterion: torch.nn.Module = ZeroInflatedNegativeBinomial
    
    # adam optimizer
    optimizer: torch.optim.Optimizer = torch.optim.Adam
    
def train_crispAI(training_config: TrainingConfig, X_trains: Dict[int, Dict[str, torch.Tensor]], y_trains: Dict[int, torch.Tensor]) -> NeuralNetRegressor:
    """train crispAI model on a set of cross validation splits

    Args:
        training_config (TrainingConfig): training configuration
        X_trains (Dict[int, Dict[str, torch.Tensor]]): training data for each split, where the key is the split index, and the value is a dictionary containing the input names and tensors for the forward function
        y_trains (Dict[int, torch.Tensor]): training labels for each split, where the key is the split index

    Returns:
        NeuralNetRegressor: _description_
    """
    model = CrispAI_pi(config)
    
    net = NeuralNetRegressor(
        model,
        max_epochs=training_config.epochs,
        lr=training_config.learning_rate,
        batch_size=training_config.batch_size,
        device=training_config.device,
        callbacks=[training_config.scheduler],
        criterion=training_config.criterion,
        optimizer=training_config.optimizer,
        iterator_train__shuffle=True,
        iterator_valid__shuffle=False,
        train_split=None,
        verbose=0
    )
        
    return net

def preprocess_data(data: pd.DataFrame) -> Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, torch.Tensor]]:
    """preprocess data for crispAI model

    Args:
        data (pd.DataFrame): data with all required columns
        
    Returns:
        Tuple[Dict[int, Dict[str, torch.Tensor]], Dict[int, torch.Tensor]: training data and labels
    """
    # load the csv data
    df = pd.read_csv(data)
    
    # split the data into folds and test data 
    
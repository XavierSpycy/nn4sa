from typing import Union, Callable, Literal, Dict

import torch

class EarlyStopping:
    def __init__(self,
                 criterion: Union[str, Callable]='loss',
                 is_greater_better: bool=False,
                 patience: int=5,
                 atol: Union[float, int]=0.0) -> None:
        
        if not isinstance(is_greater_better, bool):
            raise ValueError('is_greater_better must be a boolean')
        if not isinstance(patience, int):
            raise ValueError('patience must be an integer')
        if atol < 0.0:
            raise ValueError('atol must be a non-negative value')
        
        self.criterion = criterion
        self.is_greater_better = is_greater_better
        self.patience = patience
        self.atol = atol

        self.counter = 0
        self.best_score = float('-inf') if is_greater_better else float('inf')
        self.best_epoch = None
    
    def __call__(self, score: float, epoch: int) -> bool:
        if self.is_greater_better:
            improvement = (score - self.atol) > self.best_score
        else:
            improvement = (score + self.atol) < self.best_score
        
        if improvement:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

class History:
    def __init__(self, 
                 metrics: Dict[str, Callable]) -> None:
        
        if 'loss' in metrics:
            raise ValueError('loss is a reserved keyword and should not be included in metrics')
        
        self.metric_map = metrics
        self.metric_map['loss'] = None
        self.metric_tracker = {}
        self.metric_tracker['train_loss'] = []
        self.metric_tracker['eval_loss'] = []

        for key, value in metrics.items():
            if key != 'loss' and not callable(value):
                raise ValueError(f'{key} must be a callable')
            
            self.metric_tracker[f'train_{key}'] = []
            self.metric_tracker[f'eval_{key}'] = []

    def update(self, 
                metric: str,
                value: float,
                stage: Literal['train', 'eval']='train') -> None:

        if stage not in ['train', 'eval']:
            raise ValueError('stage must be "train" or "eval"')
        
        if metric not in self.metric_map:
            raise ValueError(f'{metric} is not a valid metric')
        
        self.metric_tracker[f'{stage}_{metric}'].append(value)
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.nn.modules.loss import _Loss
from tqdm import tqdm

from .callbacks import EarlyStopping, History

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(model: nn.Module, 
          device: torch.device,
          train_dataloader: DataLoader, 
          eval_dataloader: Optional[DataLoader]=None, 
          optimizer: Optional[Optimizer]=None, 
          criterion: _Loss=nn.CrossEntropyLoss(), 
          scheduler: Optional[_LRScheduler]=None,
          num_epochs: int=10,
          is_amp: bool=True,
          early_stopping: Optional[EarlyStopping]=None,
          history: Optional[History]=None):
    
    if optimizer is None:
        optimizer = AdamW(model.parameters())

    if early_stopping is not None and not isinstance(early_stopping, EarlyStopping):
        raise ValueError('early_stopping must be an instance of EarlyStopping')
    if history is not None and not isinstance(history, History):
        raise ValueError('history must be an instance of History')
    
    scaler = GradScaler() if is_amp else None
    model.to(device)
    model.train()

    for epoch in tqdm(range(num_epochs)):
        tqdm.write(f'Epoch {epoch + 1}/{num_epochs}')
        model.train()
        train_loss = 0.0
        eval_loss = 0.0
        train_temp_metrics = {key: 0.0 for key in history.metric_map.keys() if key != 'loss'} if history is not None else {}
        eval_temp_metrics = {key: 0.0 for key in history.metric_map.keys() if key != 'loss'} if history is not None else {}
        train_samples = 0
        eval_samples = 0

        tqdm.write('Training stage...')
        for inputs, labels in tqdm(train_dataloader, leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            batch_size = labels.size(0)
            train_samples += batch_size

            with autocast(enabled=is_amp):
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            if is_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * batch_size
            if scheduler is not None and not isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step()

            if history is not None:
                for key, metric in history.metric_map.items():
                    if key != 'loss':
                        train_temp_metrics[key] += metric(labels, predicted) * batch_size

        train_loss /= train_samples
        if history:
            history.update('loss', train_loss, stage='train')
            for key in train_temp_metrics.keys():
                train_temp_metrics[key] /= train_samples
                history.update(key, train_temp_metrics[key], stage='train')

        if eval_dataloader is not None:
            tqdm.write('Evaluating stage...')
            model.eval()
            with torch.no_grad():
                for inputs, labels in tqdm(eval_dataloader, leave=False):
                    inputs, labels = inputs.to(device), labels.to(device)
                    batch_size = labels.size(0)
                    eval_samples += batch_size
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    eval_loss += loss.item() * batch_size
                    if history is not None:
                        for key, metric in history.metric_map.items():
                            if key != 'loss':
                                eval_temp_metrics[key] += metric(labels, predicted) * batch_size

                eval_loss /= eval_samples
                if history:
                    history.update('loss', eval_loss, stage='eval')
                    for key in eval_temp_metrics.keys():
                        eval_temp_metrics[key] /= eval_samples
                        history.update(key, eval_temp_metrics[key], stage='eval')

                if early_stopping and early_stopping(eval_loss, epoch+1):
                    tqdm.write(f'Early stopping at epoch {epoch + 1}. Best epoch: {early_stopping.best_epoch}\n')
                    break
                    
        if eval_dataloader is not None:
            monitor_metric = next(iter(history.metric_map))
            tqdm.write(f'train_loss: {train_loss} - eval_loss: {eval_loss} - train_{monitor_metric}: {train_temp_metrics[monitor_metric]}, eval_{monitor_metric}: {eval_temp_metrics[monitor_metric]}')
        else:
            monitor_metric = next(iter(history.metric_map))
            tqdm.write(f'train_loss: {train_loss} - train_{monitor_metric}: {train_temp_metrics[monitor_metric]}')

    return history.metric_tracker if history else None
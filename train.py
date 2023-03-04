import torch

from typing import List, Optional, Any, Tuple
from torch import nn
from torch.utils.data import DataLoader
from sacrebleu.metrics import BLEU
from tqdm.notebook import tqdm

from model import LanguageModel


def training_epoch(model: LanguageModel, optimizer: torch.optim.Optimizer, criterion: nn.Module,
                   loader: DataLoader, tqdm_desc: str) -> float:
    """
    Process one training epoch
    :param model: language model to train
    :param optimizer: optimizer instance
    :param criterion: loss function class
    :param loader: training dataloader
    :param tqdm_desc: progress bar description
    :return: running train loss
    """
    device = next(model.parameters()).device
    train_loss = 0.0

    model.train()
    for indices_de, len_de, indices_en, len_en in tqdm(loader, desc=tqdm_desc):
        optimizer.zero_grad()

        indices_de = indices_de.to(device)
        indices_en = indices_en.to(device)
        logits = model(indices_de, len_de, indices_en, len_en, 0.7)

        loss = criterion(logits.reshape((-1, logits.shape[2])), indices_en[:, 1:len_en.max()].reshape(-1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * indices_en.shape[0]

    train_loss /= len(loader.dataset)
    return train_loss


@torch.no_grad()
def validation_epoch(model: LanguageModel, criterion: nn.Module,
                     loader: DataLoader, tqdm_desc: str) -> Tuple[float, float]:
    """
    Process one validation epoch
    :param model: language model to validate
    :param criterion: loss function class
    :param loader: validation dataloader
    :param tqdm_desc: progress bar description
    :return: validation loss and blue score
    """
    device = next(model.parameters()).device
    val_loss = 0.0

    preds = []
    texts = []

    model.eval()
    for indices_de, len_de, indices_en, len_en in tqdm(loader, desc=tqdm_desc):
        indices_de = indices_de.to(device)
        indices_en = indices_en.to(device)

        logits = model(indices_de, len_de, indices_en, len_en, 0.)
        loss = criterion(logits.reshape((-1, logits.shape[2])), indices_en[:, 1:len_en.max()].reshape(-1))

        val_loss += loss.item() * indices_en.shape[0]

        inp_texts, out_texts = loader.dataset.ids2text(indices_de, indices_en)
        texts += out_texts
        preds += [model.inference(text, 2) for text in inp_texts]

    bleu_score = BLEU().corpus_score(preds, texts).score
    val_loss /= len(loader.dataset)
    return val_loss, bleu_score


def train(model: LanguageModel, optimizer: torch.optim.Optimizer, scheduler: Optional[Any],
          train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
    """
    Train language model for several epochs
    :param model: language model to train
    :param optimizer: optimizer instance
    :param scheduler: optional scheduler
    :param train_loader: training dataloader
    :param val_loader: validation dataloader
    :param num_epochs: number of training epochs
    """
    train_losses, val_losses, val_blues = [], [], []
    criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.second.pad_id)

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        val_loss, val_blue = validation_epoch(
            model, criterion, val_loader,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        # wandb.log({'train_loss': train_loss,
        #            'val_loss': val_loss,
        #            'val_blue': val_blue})

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        val_losses += [val_loss]
        val_blues += [val_blue]

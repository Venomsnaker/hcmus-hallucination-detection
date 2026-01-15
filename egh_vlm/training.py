import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc

from egh_vlm.hallucination_detector import DetectorModule, PairedDetectorModule


def train_detector(detector: DetectorModule, loss_function, optim, data_loader: DataLoader):
    total_loss = 0

    for batch_idx, batch in enumerate(data_loader):
        optim.zero_grad()

        id, emb, grad, label = batch
        label = label.float()
        output = detector(emb, grad).squeeze(-1)
        loss = loss_function(output, label)
        loss.backward()
        optim.step()
        total_loss += loss
    return total_loss

def train_paired_detector(detector: PairedDetectorModule, loss_function, optim, data_loader: DataLoader):
    total_loss = 0

    for batch_idx, batch in enumerate(data_loader):
        optim.zero_grad()

        id, emb1, grad1, emb2, grad2, label = batch
        label = label.float()
        output = detector([[emb1, grad1], [emb2, grad2]]).squeeze()
        loss = loss_function(output, label)
        loss.backward()
        optim.step()
        total_loss += loss
    return total_loss

def eval_detector(detector: DetectorModule, data_loader):
    total_label, total_pred, total_out = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            id, emb, grad, label = batch
            output = detector(emb, grad).squeeze()
            output = torch.nan_to_num(output, nan=1.0, posinf=1.0, neginf=0.0)
            total_out += output.tolist()
            total_label += label.tolist()
            pred = list(map(lambda x: round(x), output.tolist()))
            total_pred += pred
        f1 = f1_score(total_label, total_pred)
        acc = accuracy_score(total_label, total_pred)
        precision, recall, cm = precision_recall_curve(total_label, total_pred)
        pr_auc = auc(recall, precision)
    return acc, f1, pr_auc

def eval_paired_detector(detector: PairedDetectorModule, data_loader: DataLoader):
    total_label, total_pred, total_out = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            id, emb1, grad1, emb2, grad2, label = batch
            output = detector([[emb1, grad1], [emb2, grad2]]).squeeze()
            output = torch.nan_to_num(output, nan=1.0, posinf=1.0, neginf=0.0)
            total_out += output.tolist()
            total_label += label.tolist()
            pred = list(map(lambda x: round(x), output.tolist()))
            total_pred += pred
        f1 = f1_score(total_label, total_pred)
        acc = accuracy_score(total_label, total_pred)
        precision, recall, cm = precision_recall_curve(total_label, total_pred)
        pr_auc = auc(recall, precision)
    return acc, f1, pr_auc






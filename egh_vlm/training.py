import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc

from egh_vlm.hallucination_detector import DetectorModule


def train_detector(detector: DetectorModule, loss_function, optim, data_loader: DataLoader):
    total_loss = 0

    for batch_idx, batch in enumerate(data_loader):
        optim.zero_grad()
        # Get the inputs
        id, emb, grad, label = batch
        label = label.float()
        # Forward + backward + optimize
        output = detector(emb, grad).squeeze(-1)
        loss = loss_function(output, label)
        loss.backward()
        optim.step()
        total_loss += loss
    return total_loss

def eval_detector(detector: DetectorModule, data_loader: DataLoader):
    total_label, total_pred = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            # Get the inputs
            id, emb, grad, label = batch
            label = label.float()
            total_label += label.tolist()
            # Forward pass
            output = detector(emb, grad).squeeze(-1)
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
            # Get the binary predictions
            pred = list(map(lambda x: round(x), output.tolist()))
            total_pred += pred
        acc = accuracy_score(total_label, total_pred)
        f1 = f1_score(total_label, total_pred)
        precision, recall, cm = precision_recall_curve(total_label, total_pred)
        pr_auc = auc(recall, precision)
    return acc, f1, pr_auc




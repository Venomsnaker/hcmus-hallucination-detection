import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc


def get_mean(input_list: list):
    temp = [torch.mean(x, dim=0).squeeze(0) for x in input_list]
    return torch.stack(temp).to(temp[0].device)

class HalluFFNDetector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, w: float=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.w = w

    def forward(self, emb: torch.Tensor, grad: torch.Tensor):
        emb = get_mean(emb)
        grad = get_mean(grad)

        x = self.w * emb + (1 - self.w) * grad
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)

def train_ffn_detector(detector: HalluFFNDetector, loss_fn: nn.Module, optim: torch.optim.Optimizer, data_loader: DataLoader):
    total_loss = 0

    for batch_idx, batch in enumerate(data_loader):
        optim.zero_grad()
        id, emb, grad, label = batch
        label = label.float()
        output = detector(emb, grad).squeeze(-1)
        loss = loss_fn(output, label)
        loss.backward()
        optim.step()
        total_loss += loss
    return total_loss

def eval_ffn_detector(detector: HalluFFNDetector, data_loader: DataLoader):
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
    return {
        'accuracy': acc,
        'f1_score': f1,
        'pr_auc': pr_auc
    } 
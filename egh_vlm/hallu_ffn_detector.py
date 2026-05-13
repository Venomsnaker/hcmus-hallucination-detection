import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc


def get_mean(input_list: list):
    temp = [torch.mean(x, dim=0).squeeze(0) for x in input_list]
    return torch.stack(temp).to(temp[0].device)

class HalluFFNDetector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        device: torch.device,
        w: float = 0.5,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.fc1 = nn.Linear(input_dim, hidden_dim, dtype=self.dtype, device=self.device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=self.dtype, device=self.device)
        self.fc3 = nn.Linear(hidden_dim, output_dim, dtype=self.dtype, device=self.device)
        self.w = w

    def forward(self, emb: torch.Tensor, grad: torch.Tensor):
        emb = get_mean(emb)
        grad = get_mean(grad)

        # Keep feature tensors aligned with model params
        emb = emb.to(device=self.device, dtype=self.dtype)
        grad = grad.to(device=self.device, dtype=self.dtype)

        x = self.w * emb + (1 - self.w) * grad
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)

def train_ffn_detector(detector: HalluFFNDetector, loss_fn: nn.Module, optim: torch.optim.Optimizer, data_loader: DataLoader):
    total_loss = 0.0
    eps = 1e-6

    for batch_idx, batch in enumerate(data_loader):
        optim.zero_grad()
        id, emb, grad, label = batch
        
        output = detector(emb, grad).squeeze(-1)
        label = label.to(device=output.device, dtype=output.dtype)
        # Stabilize output & label
        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0).clamp(eps, 1.0 - eps)
        label = torch.nan_to_num(label, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        
        loss = loss_fn(output, label)
        loss.backward()
        optim.step()
        total_loss += loss.item()
    return total_loss

def eval_ffn_detector(detector: HalluFFNDetector, data_loader: DataLoader):
    total_label, total_pred, total_score = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            id, emb, grad, label = batch
            
            # Forward pass
            output = detector(emb, grad).squeeze(-1)
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            
            # Collect pred and label
            total_label += label.cpu().tolist()
            total_pred += list(map(lambda x: round(x), output.cpu().tolist()))
            total_score += output.cpu().tolist()
        acc = accuracy_score(total_label, total_pred)
        f1 = f1_score(total_label, total_pred)
        precision, recall, cm = precision_recall_curve(total_label, total_score)
        pr_auc = auc(recall, precision)
    return {
        'acc': acc,
        'f1': f1,
        'pr_auc': pr_auc
    } 
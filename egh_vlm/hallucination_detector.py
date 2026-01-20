import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mean(input_list):
    temp = [torch.mean(x, dim=0).squeeze(0) for x in input_list]
    return torch.stack(temp).to(temp[0].device)

class DetectorModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, w=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.w = w

    def forward(self, emb, grad):
        emb = get_mean(emb)
        grad = get_mean(grad)

        x = self.w * emb + (1 - self.w) * grad
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)

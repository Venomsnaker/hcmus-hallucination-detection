import torch
import torch.nn as nn
import torch.nn.functional as F


def get_mean(input_list):
    temp = []

    for x in input_list:
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        temp.append(torch.mean(x, dim=0).squeeze(0))
    return torch.stack(temp).to(temp[0].device)

class DetectorModule(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
    ):
        super().__init__()
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.mix_logits = nn.Parameter(torch.zeros(4))

    def forward(self, qa_embedding, qa_gradient, ia_embedding, ia_gradient):
        qa_embedding = get_mean(qa_embedding)
        qa_gradient = get_mean(qa_gradient)
        ia_embedding = get_mean(ia_embedding)
        ia_gradient = get_mean(ia_gradient)
        weights = torch.softmax(self.mix_logits, dim=0)

        x = (weights[0] * qa_embedding +
            weights[1] * qa_gradient +
            weights[2] * ia_embedding +
            weights[3] * ia_gradient)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

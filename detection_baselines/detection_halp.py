import gc
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, auc, f1_score,
                              precision_recall_curve, precision_score,
                              recall_score)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from egh_vlm.utils import Qwen3ModelBundle


@dataclass
class HalpFeatures:
    hidden: torch.Tensor  # (hidden_dim,) mean-pooled context representation


class HalpDataset(Dataset):
    def __init__(self, ids=None, hiddens=None, labels=None):
        self.ids = ids if ids is not None else []
        self.hiddens = hiddens if hiddens is not None else []
        self.labels = labels if labels is not None else []

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        return self.ids[idx], self.hiddens[idx], self.labels[idx]

    def get_by_id(self, id: str):
        try:
            return self.__getitem__(self.ids.index(id))
        except ValueError:
            return None

    def add_item(self, id: str, hidden: torch.Tensor, label: int):
        self.ids.append(id)
        self.hiddens.append(hidden.squeeze())
        self.labels.append(label)


def halp_collate_fn(batch):
    ids, hiddens, labels = [], [], []
    for item in batch:
        ids.append(item[0])
        hiddens.append(item[1])
        labels.append(item[2])
    return ids, torch.stack(hiddens), torch.tensor(labels)


def save_halp_dataset(dataset, path):
    torch.save({
        'ids': dataset.ids,
        'hiddens': dataset.hiddens,
        'labels': dataset.labels
    }, path)


def load_halp_dataset(path):
    ckpt = torch.load(path, map_location='cpu')
    return HalpDataset(ckpt['ids'], ckpt['hiddens'], ckpt['labels'])


def extract_halp_qwen3(
    context_messages: list,
    model_bundle: Qwen3ModelBundle,
    targeted_layer: int = -1) -> HalpFeatures:
    """
    Single forward pass on context only (no answer required).
    Mean-pools hidden states at targeted_layer — the representation the model
    holds just before it would begin generating.
    """
    model, processor, device = model_bundle.model, model_bundle.processor, model_bundle.device

    ids = processor.apply_chat_template(
        context_messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors='pt'
    )
    ids = {k: v.to(device) for k, v in ids.items()}

    with torch.no_grad():
        model.eval()
        output = model(**ids, output_hidden_states=True)
        hidden = output.hidden_states[targeted_layer].squeeze(0)  # (seq_len, hidden_dim)
        hidden_pooled = hidden.mean(dim=0)                        # (hidden_dim,)

    result = HalpFeatures(hidden=hidden_pooled.detach().to('cpu'))
    del ids, output, hidden, hidden_pooled
    return result


def extract_halp(
    dataset,
    model_bundle,
    client_type: str = 'qwen3',
    save_path: str = None,
    save_interval: int = 20,
    mask_mode: str = None,
    targeted_layer: int = -1):
    """
    client_type : 'qwen3'
    mask_mode   : None | 'image' | 'question'
    targeted_layer : layer index for hidden state extraction
    """
    if client_type not in ['qwen3']:
        print('Unsupported client')
        return None

    if mask_mode not in [None, 'image', 'question']:
        print('Incorrect mask mode')
        return None

    processed = HalpDataset()
    if save_path is not None and os.path.exists(save_path):
        processed = load_halp_dataset(save_path)
    processed_ids = set(processed.ids)

    for item in tqdm(dataset, desc=f'Extracting HALP features ({client_type})'):
        if item['id'] in processed_ids:
            continue

        context = []
        if item['image_path'] is not None and mask_mode != 'image':
            context.append({'type': 'image', 'image': item['image_path']})
        if item['question'] is not None and mask_mode != 'question':
            context.append({'type': 'text', 'text': item['question']})

        features = extract_halp_qwen3(
            context_messages=[{'role': 'user', 'content': context}],
            model_bundle=model_bundle,
            targeted_layer=targeted_layer
        )
        hidden = features.hidden

        if hidden.numel() == 0 or torch.isnan(hidden).any() or torch.isinf(hidden).any():
            print(f"Skipping id={item['id']}: invalid features.")
            continue

        processed.add_item(item['id'], hidden, item['label'])

        if save_path is not None and len(processed) % save_interval == 0:
            save_halp_dataset(processed, save_path)

        del features, hidden, context
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if save_path is not None:
        save_halp_dataset(processed, save_path)
    return processed


class HalpLinearProbe(nn.Module):
    """Single linear layer (logistic regression) trained on mean-pooled context hidden states."""

    def __init__(self, input_dim: int, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.fc = nn.Linear(input_dim, 1, dtype=dtype, device=device)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.to(device=self.device, dtype=self.dtype)
        return torch.sigmoid(self.fc(hidden))


def train_halp_probe(
    probe: HalpLinearProbe,
    loss_fn: nn.Module,
    optim: torch.optim.Optimizer,
    data_loader: DataLoader):

    eps = 1e-6
    total_loss = 0.0

    for _, hidden, label in data_loader:
        optim.zero_grad()
        output = probe(hidden).squeeze(-1)
        label = label.to(device=output.device, dtype=output.dtype)
        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0).clamp(eps, 1.0 - eps)
        loss = loss_fn(output, label.clamp(0.0, 1.0))
        loss.backward()
        optim.step()
        total_loss += loss.item()

    return total_loss


def eval_halp_probe(probe: HalpLinearProbe, data_loader: DataLoader):
    total_label, total_pred, total_score = [], [], []

    with torch.no_grad():
        for _, hidden, label in data_loader:
            output = probe(hidden).squeeze(-1)
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            total_label += label.tolist()
            total_pred += [round(x) for x in output.cpu().tolist()]
            total_score += output.cpu().tolist()

    precision, recall, _ = precision_recall_curve(total_label, total_score)
    return {
        'acc': accuracy_score(total_label, total_pred),
        'f1': f1_score(total_label, total_pred, zero_division=0),
        'precision': precision_score(total_label, total_pred, zero_division=0),
        'recall': recall_score(total_label, total_pred, zero_division=0),
        'pr_auc': auc(recall, precision)
    }

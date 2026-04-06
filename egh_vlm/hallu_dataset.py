import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


class HalluDataset(Dataset):
    def __init__(self, ids=[], embs=[], grads=[], labels=[]):
        self.ids=ids
        self.embs=embs
        self.grads=grads
        self.labels=labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        return self.ids[idx], self.embs[idx], self.grads[idx], self.labels[idx]
    
    def get_by_id(self, id: str):
        try:
            idx = self.ids.index(id)
            return self.__getitem__(idx)
        except ValueError:
            return None
    
    def add_item(self, id: str, emb: torch.Tensor, grad: torch.Tensor, label: int):
        self.ids.append(id)
        self.embs.append(emb)
        self.grads.append(grad)
        self.labels.append(label)

def hallu_collate_fn(batch):
    ids, embs, grads, labels = [], [], [], []

    for item in batch:
        ids.append(item[0])
        embs.append(item[1])
        grads.append(item[2])
        labels.append(item[3])
    return ids, embs, grads, torch.tensor(labels)

def save_hallu_dataset(dataset, path):
    torch.save({
        'ids': dataset.ids,
        'embs': dataset.embs,
        'grads': dataset.grads,
        'labels': dataset.labels
    }, path)

def load_hallu_dataset(path):
    checkpoint = torch.load(path, map_location='cpu')
    return HalluDataset(
        checkpoint['ids'],
        checkpoint['embs'],
        checkpoint['grads'],
        checkpoint['labels'])

def split_stratified(dataset, train_ratio=0.7, random_state=42):
    labels = np.array(dataset.labels)

    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=1-train_ratio,
        stratify=labels,
        random_state=random_state
    )
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    return train_dataset, test_dataset

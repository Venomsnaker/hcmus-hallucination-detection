import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


class HallucinationDataset(Dataset):
    def __init__(self, ids=[], embs=[], grads=[], labels=[]):
        self.ids = ids
        self.embs = embs
        self.grads = grads
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.embs[idx], self.grads[idx], self.labels[idx]
    
    def add_item(self, id, emb, grad, label):
        self.ids.append(id)
        self.embs.append(emb)
        self.grads.append(grad)
        self.labels.append(label)

def hallucination_collate_fn(batch):
    ids, embs, grads, labels = [], [], [], []

    for sample in batch:
        ids.append(sample[0])
        embs.append(sample[1])
        grads.append(sample[2])
        labels.append(sample[3])
    return ids, embs, grads, torch.tensor(labels)

class PairedHallucinationDataset(Dataset):
    def __init__(self, ids=[], embs1=[], grads1=[], embs2=[], grads2 = [], labels=[]):
        self.ids = ids
        self.embs1 = embs1
        self.grads1 = grads1
        self.embs2 = embs2
        self.grads2 = grads2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.embs1[idx], self.grads1[idx], self.embs2[idx], self.grads2[idx], self.labels[idx]
    
    def add_item(self, id, feats, label):
        '''
        feats: [[emb1, grad1], [emb2, grad2]]
        '''
        self.ids.append(id)
        self.embs1.append(feats[0][0])
        self.grads1.append(feats[0][1])
        self.embs2.append(feats[1][0])
        self.grads2.append(feats[1][1])
        self.labels.append(label)

def paired_hallucination_collate_fn(batch):
    ids, embs1, grads1, embs2, grads2, labels = [], [], [], [], [], []

    for sample in batch:
        ids.append(sample[0])
        embs1.append(sample[1])
        grads1.append(sample[2])
        embs2.append(sample[3])
        grads2.append(sample[4])
        labels.append(sample[5])
    return ids, embs1, grads1, embs2, grads2, torch.tensor(labels)

def save_features(dataset, path):
    torch.save({
        'ids': dataset.ids,
        'embs': dataset.embs,
        'grads': dataset.grads,
        'labels': dataset.labels
    }, path)

def load_features(path):
    checkpoint = torch.load(path, map_location='cpu')
    return HallucinationDataset(
        checkpoint['ids'],
        checkpoint['embs'],
        checkpoint['grads'],
        checkpoint['labels']
    )

def split_stratified(dataset, train_ratio=0.7, random_state=42):
    labels = np.array(dataset.labels)

    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=1 - train_ratio,
        stratify=labels,
        random_state=random_state
    )
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    return train_dataset, test_dataset

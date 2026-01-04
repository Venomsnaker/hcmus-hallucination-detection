import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset


class HallucinationDataset(Dataset):
    def __init__(self, ids=[], embeddings=[], gradients=[], labels=[]):
        self.ids = ids
        self.embeddings = embeddings
        self.gradients = gradients
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.embeddings[idx], self.gradients[idx], self.labels[idx]
    
    def add_item(self, id, embedding, gradient, label):
        self.ids.append(id)
        self.embeddings.append(embedding)
        self.gradients.append(gradient)
        self.labels.append(label)

def save_features(dataset, path):
    torch.save({
        'ids': dataset.ids,
        'embeddings': dataset.embeddings,
        'gradients': dataset.gradients,
        'labels': dataset.labels
    }, path)

def load_features(path):
    checkpoint = torch.load(path, map_location='cpu')
    return HallucinationDataset(
        checkpoint['ids'],
        checkpoint['embeddings'],
        checkpoint['gradients'],
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

def hallucination_collate_fn(batch):
    ids, embeddings, gradients, labels = [], [], [], []

    for sample in batch:
        ids.append(sample[0])
        embeddings.append(sample[1])
        gradients.append(sample[2])
        labels.append(sample[3])
    return ids, embeddings, gradients, torch.tensor(labels)
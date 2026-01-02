import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split

class HallucinationDataset(Dataset):
    def __init__(self, embeddings, gradients, labels):
        self.embeddings = embeddings
        self.gradients = gradients
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.gradients[idx], self.labels[idx]

def save_features(dataset, path):
    torch.save({
        'embeddings': dataset.embeddings,
        'gradients': dataset.gradients,
        'labels': dataset.labels
    }, path)

def load_features(path):
    checkpoint = torch.load(path, map_location='cpu')
    return HallucinationDataset(
        checkpoint['embeddings'],
        checkpoint['gradients'],
        checkpoint['labels']
    )


def stratified_split(dataset, train_ratio=0.7, random_state=42):
    labels = dataset.labels
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.array(labels)
    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=1 - train_ratio,
        stratify=labels_np,
        random_state=random_state
    )
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    return train_dataset, val_dataset

def hallucination_collate_fn(batch):
    embeddings = []
    gradients = []
    labels = []

    for sample in batch:
        embeddings.append(sample[0])
        gradients.append(sample[1])
        labels.append(sample[2])
    return embeddings, gradients,torch.tensor(labels)
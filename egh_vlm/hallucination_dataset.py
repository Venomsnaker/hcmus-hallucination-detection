import torch
from torch.utils.data import Dataset


class HallucinationDataset(Dataset):
    def __init__(self, qa_embeddings, qa_gradients, ia_embeddings, ia_gradients, labels):
        self.qa_embeddings = qa_embeddings
        self.qa_gradients = qa_gradients
        self.ia_embeddings = ia_embeddings
        self.ia_gradients = ia_gradients
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.qa_embeddings[idx], self.qa_gradients[idx], self.ia_embeddings[idx], self.ia_gradients[idx], self.labels[idx]


def hallucination_collate_fn(batch):
    qa_embeddings = []
    qa_gradients = []
    ia_embeddings = []
    ia_gradients = []
    labels = []

    for sample in batch:
        qa_embeddings.append(sample[0])
        qa_gradients.append(sample[1])
        ia_embeddings.append(sample[2])
        ia_gradients.append(sample[3])
        labels.append(sample[4])
    return qa_embeddings, qa_gradients, ia_embeddings, ia_gradients, torch.tensor(labels)
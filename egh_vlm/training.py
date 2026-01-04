import os
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_curve, auc

from egh_vlm.extract_feature import batch_extract_features
from egh_vlm.hallucination_dataset import load_features, save_features
from egh_vlm.utils import load_hallusion_bench_dataset


def get_features(saved_path, dataset_dir, model, processor, device, sample_size=None):
    if os.path.isfile(saved_path):
        features = load_features(saved_path)
        return features, features.embeddings[0].size(-1) if len(features.embeddings) > 0 else 0
    else:
        data_list = load_hallusion_bench_dataset(dataset_dir, sample_size=sample_size)
        features = batch_extract_features(data_list, model, processor, device, saved_path=saved_path)
        save_features(features, saved_path)
        return features, features.embeddings[0].size(-1) if len(features.embeddings) > 0 else 0

def train(detector, loss_function, optimizer, data_loader):
    total_loss = 0

    for _, batch in enumerate(data_loader):
        optimizer.zero_grad()
        embedding, gradient, label = batch
        label = label.float()
        output = detector(embedding, gradient).squeeze()
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss

def eval_detector(detector, data_loader):
    total_label, total_pred, total_out = [], [], []

    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            embedding, gradient, label = batch

            output = detector(embedding, gradient).squeeze()
            output = torch.nan_to_num(output, nan=1.0, posinf=1.0, neginf=0.0)
            total_out += output.tolist()
            total_label += label.tolist()
            pred = list(map(lambda x: round(x), output.tolist()))
            total_pred += pred
        f1 = f1_score(total_label, total_pred)
        acc = accuracy_score(total_label, total_pred)
        precision, recall, cm = precision_recall_curve(total_label, total_pred)
        pr_auc = auc(recall, precision)
    return acc, f1, pr_auc





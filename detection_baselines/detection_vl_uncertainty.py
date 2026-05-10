import gc
import math
import os
import random
from dataclasses import dataclass

import torch
from PIL import Image, ImageEnhance
from sklearn.metrics import (accuracy_score, auc, f1_score,
                              precision_recall_curve, precision_score,
                              recall_score)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from egh_vlm.utils import Qwen3ModelBundle, get_pred, get_response_qwen3


def _augment_image(image: Image.Image, seed: int) -> Image.Image:
    """Deterministic image augmentation: brightness, contrast, random crop."""
    rng = random.Random(seed)
    img = image.copy()

    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.75, 1.25))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.80, 1.20))

    w, h = img.size
    scale = rng.uniform(0.88, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    left = rng.randint(0, w - nw)
    top = rng.randint(0, h - nh)
    img = img.crop((left, top, left + nw, top + nh)).resize((w, h), Image.BILINEAR)

    return img


@dataclass
class UncertaintyFeatures:
    score: torch.Tensor  # scalar: binary entropy of yes/no predictions across augmented runs


class UncertaintyDataset(Dataset):
    def __init__(self, ids=None, scores=None, labels=None):
        self.ids = ids if ids is not None else []
        self.scores = scores if scores is not None else []
        self.labels = labels if labels is not None else []

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        return self.ids[idx], self.scores[idx], self.labels[idx]

    def get_by_id(self, id: str):
        try:
            return self.__getitem__(self.ids.index(id))
        except ValueError:
            return None

    def add_item(self, id: str, score: torch.Tensor, label: int):
        self.ids.append(id)
        self.scores.append(score.squeeze())
        self.labels.append(label)


def uncertainty_collate_fn(batch):
    ids, scores, labels = [], [], []
    for item in batch:
        ids.append(item[0])
        scores.append(item[1])
        labels.append(item[2])
    return ids, torch.stack(scores), torch.tensor(labels)


def save_uncertainty_dataset(dataset, path):
    torch.save({'ids': dataset.ids, 'scores': dataset.scores, 'labels': dataset.labels}, path)


def load_uncertainty_dataset(path):
    ckpt = torch.load(path, map_location='cpu')
    return UncertaintyDataset(ckpt['ids'], ckpt['scores'], ckpt['labels'])


def extract_uncertainty_qwen3(
    image_path: str,
    question: str,
    model_bundle: Qwen3ModelBundle,
    n_perturbations: int = 5) -> UncertaintyFeatures | None:
    """
    Runs n_perturbations visually-augmented copies of the image through the model.
    Returns binary entropy of yes/no predictions as the uncertainty score.

    High entropy (near 1.0) → inconsistent answers → likely hallucination.
    Low entropy (near 0.0) → consistent answers → reliable response.
    """
    base_image = Image.open(image_path).convert('RGB')
    predictions = []

    for i in range(n_perturbations):
        aug_image = _augment_image(base_image, seed=i)
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': aug_image},
            {'type': 'text', 'text': question + ' Answer with yes or no.'}
        ]}]
        response = get_response_qwen3(messages, model_bundle, max_new_tokens=8)
        pred = get_pred(response)
        if pred != 0.5:
            predictions.append(float(pred))

    if not predictions:
        return None

    p = sum(predictions) / len(predictions)
    entropy = 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    return UncertaintyFeatures(score=torch.tensor([entropy], dtype=torch.float32))


def extract_uncertainty(
    dataset,
    model_bundle,
    client_type: str = 'qwen3',
    save_path: str = None,
    save_interval: int = 20,
    n_perturbations: int = 5):
    """
    client_type    : 'qwen3'
    n_perturbations: number of visual augmentations per sample (default 5)

    Note: n_perturbations × dataset_size inference runs — budget accordingly.
    Skips samples with no image_path or where all perturbations return ambiguous answers.
    """
    if client_type not in ['qwen3']:
        print('Unsupported client')
        return None

    processed = UncertaintyDataset()
    if save_path is not None and os.path.exists(save_path):
        processed = load_uncertainty_dataset(save_path)
    processed_ids = set(processed.ids)

    for item in tqdm(dataset, desc=f'Extracting VL-Uncertainty features ({client_type})'):
        if item['id'] in processed_ids:
            continue
        if item['image_path'] is None:
            print(f"Skipping id={item['id']}: no image path.")
            continue

        features = extract_uncertainty_qwen3(
            image_path=item['image_path'],
            question=item['question'],
            model_bundle=model_bundle,
            n_perturbations=n_perturbations
        )

        if features is None:
            print(f"Skipping id={item['id']}: all perturbations returned ambiguous answers.")
            continue

        score = features.score
        if score.numel() == 0 or torch.isnan(score).any() or torch.isinf(score).any():
            print(f"Skipping id={item['id']}: invalid score.")
            continue

        processed.add_item(item['id'], score, item['label'])

        if save_path is not None and len(processed) % save_interval == 0:
            save_uncertainty_dataset(processed, save_path)

        del features, score
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if save_path is not None:
        save_uncertainty_dataset(processed, save_path)
    return processed


class UncertaintyThresholdDetector:
    def __init__(self, threshold: float = None):
        self.threshold = threshold

    def score(self, uncertainty: torch.Tensor) -> torch.Tensor:
        return uncertainty.float()

    def predict(self, uncertainty: torch.Tensor) -> torch.Tensor:
        if self.threshold is None:
            raise ValueError('Detector threshold is not trained.')
        return self.score(uncertainty) >= self.threshold


def train_uncertainty_detector(detector: UncertaintyThresholdDetector, data_loader: DataLoader):
    total_label, total_score = [], []

    for _, uncertainty, label in data_loader:
        score = torch.nan_to_num(uncertainty.float(), nan=0.0, posinf=1.0, neginf=0.0)
        total_label += label.tolist()
        total_score += score.squeeze(-1).tolist()

    thresholds = sorted(set(total_score))
    best_f1, best_t = -1.0, thresholds[0] if thresholds else 0.0

    for t in thresholds:
        preds = [int(s >= t) for s in total_score]
        f1 = f1_score(total_label, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    detector.threshold = float(best_t)
    return {'threshold': detector.threshold, 'f1': best_f1}


def eval_uncertainty_detector(detector: UncertaintyThresholdDetector, data_loader: DataLoader):
    total_label, total_pred, total_score = [], [], []

    for _, uncertainty, label in data_loader:
        score = torch.nan_to_num(uncertainty.float(), nan=0.0, posinf=1.0, neginf=0.0).squeeze(-1)
        pred = (score >= detector.threshold).int()
        total_label += label.tolist()
        total_pred += pred.tolist()
        total_score += score.tolist()

    precision, recall, _ = precision_recall_curve(total_label, total_score)
    return {
        'acc': accuracy_score(total_label, total_pred),
        'f1': f1_score(total_label, total_pred, zero_division=0),
        'precision': precision_score(total_label, total_pred, zero_division=0),
        'recall': recall_score(total_label, total_pred, zero_division=0),
        'pr_auc': auc(recall, precision)
    }
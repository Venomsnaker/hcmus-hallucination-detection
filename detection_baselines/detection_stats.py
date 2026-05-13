import os
import gc
from dataclasses import dataclass

import torch
from sklearn.metrics import accuracy_score, auc, f1_score, precision_recall_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from egh_vlm.utils import Qwen3ModelBundle


@dataclass
class StatsFeatures:
    entropy_mean: torch.Tensor
    entropy_max: torch.Tensor
    prob_mean: torch.Tensor
    prob_max: torch.Tensor

class StatsDataset(Dataset):
    def __init__(self, ids=None, entropy_means=None, entropy_maxs=None, prob_means=None, prob_maxs=None, labels=None):
        self.ids = ids if ids is not None else []
        self.entropy_means = entropy_means if entropy_means is not None else []
        self.entropy_maxs = entropy_maxs if entropy_maxs is not None else []
        self.prob_means = prob_means if prob_means is not None else []
        self.prob_maxs = prob_maxs if prob_maxs is not None else []
        self.labels = labels if labels is not None else []

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        return self.ids[idx], self.entropy_means[idx], self.entropy_maxs[idx], self.prob_means[idx], self.prob_maxs[idx], self.labels[idx]

    def get_by_id(self, id: str):
        try:
            idx = self.ids.index(id)
            return self.__getitem__(idx)
        except ValueError:
            return None

    def add_item(self, id: str, entropy_mean: torch.Tensor, entropy_max: torch.Tensor, prob_mean: torch.Tensor, prob_max: torch.Tensor, label: int):
        self.ids.append(id)
        self.entropy_means.append(entropy_mean.squeeze())
        self.entropy_maxs.append(entropy_max.squeeze())
        self.prob_means.append(prob_mean.squeeze())
        self.prob_maxs.append(prob_max.squeeze())
        self.labels.append(label)

def stats_collate_fn(batch):
    ids, entropy_means, entropy_maxs, prob_means, prob_maxs, labels = [], [], [], [], [], []

    for item in batch:
        ids.append(item[0])
        entropy_means.append(item[1])
        entropy_maxs.append(item[2])
        prob_means.append(item[3])
        prob_maxs.append(item[4])
        labels.append(item[5])
    return ids, torch.stack(entropy_means), torch.stack(entropy_maxs), torch.stack(prob_means), torch.stack(prob_maxs), torch.tensor(labels)

def save_stats_dataset(dataset, path):
    torch.save({
        'ids': dataset.ids,
        'entropy_means': dataset.entropy_means,
        'entropy_maxs': dataset.entropy_maxs,
        'prob_means': dataset.prob_means,
        'prob_maxs': dataset.prob_maxs,
        'labels': dataset.labels
    }, path)

def load_stats_dataset(path):
    checkpoint = torch.load(path, map_location='cpu')
    return StatsDataset(
        checkpoint['ids'],
        checkpoint['entropy_means'],
        checkpoint['entropy_maxs'],
        checkpoint['prob_means'],
        checkpoint['prob_maxs'],
        checkpoint['labels'])

def sequence_entropy(logits: torch.Tensor):
    log_prob = torch.log_softmax(logits.float(), dim=-1)
    prob = log_prob.exp()
    return -(prob * log_prob).sum(dim=-1)

def sequence_prob(logits: torch.Tensor):
    return torch.softmax(logits.float().clamp(-100, 100), dim=-1).max(dim=-1).values

def summarize_metrics(context_entropy: torch.Tensor, context_prob: torch.Tensor):
    entropy_mean = context_entropy.mean().float().unsqueeze(0)
    entropy_max = context_entropy.max().float().unsqueeze(0)
    prob_mean = context_prob.mean().float().unsqueeze(0)
    prob_max = context_prob.max().float().unsqueeze(0)
    return StatsFeatures(entropy_mean=entropy_mean, entropy_max=entropy_max, prob_mean=prob_mean, prob_max=prob_max)

def extract_stats_qwen3(
    messages: list,
    model_bundle: Qwen3ModelBundle):
    model, processor, device = model_bundle.model, model_bundle.processor, model_bundle.device

    ids = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors='pt'
    )
    ids = {k: v.to(device) for k, v in ids.items()}

    with torch.no_grad():
        model.eval()

        output = model(**ids)
        answer_token_length = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors='pt'
        )['input_ids'].shape[1]

        logits = output.logits.squeeze(0)[-answer_token_length + 1:, :]

        if logits.shape[0] == 0:
            entropy = torch.empty(0, device=device)
            stats = StatsFeatures(
                entropy_mean=torch.empty(0, device=device),
                entropy_max=torch.empty(0, device=device),
                prob_mean=torch.empty(0, device=device),
                prob_max=torch.empty(0, device=device),
            )
        else:
            entropy = sequence_entropy(logits)
            prob = sequence_prob(logits)
            stats = summarize_metrics(entropy, prob)

    features = StatsFeatures(
        entropy_mean=stats.entropy_mean.detach().to('cpu'),
        entropy_max=stats.entropy_max.detach().to('cpu'),
        prob_mean=stats.prob_mean.detach().to('cpu'),
        prob_max=stats.prob_max.detach().to('cpu'),
    )

    del ids, output, logits
    del entropy, prob, stats

    return features

def extract_stats(
    dataset: any,
    model_bundle: any,
    client_type: str = 'qwen3',
    save_path: str = None,
    save_interval: int = 20,
    mask_mode: str = None):
    if client_type not in ['qwen3']:
        print('Unsupported client')
        return None

    if mask_mode not in [None, 'image', 'question']:
        print('Incorrect mask mode')
        return None

    processed_entropies = StatsDataset()

    if save_path is not None and os.path.exists(save_path):
        processed_entropies = load_stats_dataset(save_path)
    processed_ids = set(processed_entropies.ids)

    if client_type == 'qwen3':
        for item in tqdm(dataset, desc=f'Extracting stats for client {client_type}'):
            if item['id'] in processed_ids:
                continue

            question = item['question']
            image_path = item['image_path']
            answer = item['answer']

            context = []

            if image_path is not None and mask_mode != 'image':
                context.append({'type': 'image', 'image': image_path})
            if question is not None and mask_mode != 'question':
                context.append({'type': 'text', 'text': question})

            messages = [
                {'role': 'user', 'content': context},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}
            ]

            features = extract_stats_qwen3(
                messages=messages,
                model_bundle=model_bundle,
            )
            entropy_mean = features.entropy_mean
            entropy_max = features.entropy_max
            prob_mean = features.prob_mean
            prob_max = features.prob_max

            has_non_empty_features = entropy_mean.numel() > 0 and entropy_max.numel() > 0 and prob_mean.numel() > 0 and prob_max.numel() > 0
            has_valid_values = (
                not torch.isnan(entropy_mean).any()
                and not torch.isinf(entropy_mean).any()
                and not torch.isnan(entropy_max).any()
                and not torch.isinf(entropy_max).any()
                and not torch.isnan(prob_mean).any()
                and not torch.isinf(prob_mean).any()
                and not torch.isnan(prob_max).any()
                and not torch.isinf(prob_max).any()
            )

            if has_non_empty_features and has_valid_values:
                processed_entropies.add_item(item['id'], entropy_mean, entropy_max, prob_mean, prob_max, item['label'])
            else:
                print(f"Skipping id={item['id']} due to invalid metrics.")
                continue
            
            if save_path is not None and (len(processed_entropies) % save_interval == 0):
                save_stats_dataset(processed_entropies, save_path)

            del features, entropy_mean, entropy_max, prob_mean, prob_max, messages
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if save_path is not None:
            save_stats_dataset(processed_entropies, save_path)

        return processed_entropies
    else:
        print('Unsupported client')
        return None

class ThresholdDetector:
    def __init__(self, metric_type: str = 'entropy_mean', threshold: float = None):
        if metric_type not in ['entropy_mean', 'entropy_max', 'prob_mean', 'prob_max']:
            raise ValueError("metric_type must be one of 'entropy_mean', 'entropy_max', 'prob_mean', or 'prob_max'")
        self.metric_type = metric_type
        self.threshold = threshold

    def _is_higher_score_better(self):
        return self.metric_type in ['entropy_mean', 'entropy_max']

    def score(self, entropy_mean: torch.Tensor, entropy_max: torch.Tensor, prob_mean: torch.Tensor, prob_max: torch.Tensor):
        if self.metric_type == 'entropy_mean':
            return self._reduce_scores(entropy_mean, reduce='mean')
        if self.metric_type == 'entropy_max':
            return self._reduce_scores(entropy_max, reduce='max')
        if self.metric_type == 'prob_mean':
            return self._reduce_scores(prob_mean, reduce='mean')
        return self._reduce_scores(prob_max, reduce='max')

    @staticmethod
    def _reduce_scores(score: torch.Tensor, reduce: str):
        score = score.float()
        if score.ndim <= 1:
            return score.reshape(-1)

        score = score.reshape(score.shape[0], -1)
        if reduce == 'mean':
            return score.mean(dim=1)
        if reduce == 'max':
            return score.max(dim=1).values
        raise ValueError("reduce must be 'mean' or 'max'")

    def predict(self, entropy_mean: torch.Tensor, entropy_max: torch.Tensor, prob_mean: torch.Tensor, prob_max: torch.Tensor):
        if self.threshold is None:
            raise ValueError('Detector threshold is not trained.')
        scores = self.score(entropy_mean, entropy_max, prob_mean, prob_max)
        if self._is_higher_score_better():
            return scores >= self.threshold
        return scores <= self.threshold

def train_threshold_detector(detector: ThresholdDetector, data_loader: DataLoader):
    total_label, total_score = [], []

    for batch_idx, batch in enumerate(data_loader):
        id, entropy_mean, entropy_max, prob_mean, prob_max, label = batch
        score = detector.score(entropy_mean, entropy_max, prob_mean, prob_max)
        score = torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)

        total_label += label.cpu().tolist()
        total_score += score.cpu().reshape(-1).tolist()

    if not total_score:
        detector.threshold = 0.0
        return {
            'threshold': detector.threshold,
            'f1': 0.0
        }

    sorted_scores = sorted(total_score)
    num_thresholds = min(1000, len(sorted_scores))
    if num_thresholds == 1:
        thresholds = [sorted_scores[0]]
    else:
        sampled_indices = torch.linspace(0, len(sorted_scores) - 1, steps=num_thresholds).round().long().tolist()
        thresholds = [sorted_scores[idx] for idx in sampled_indices]
        thresholds = sorted(set(thresholds))

    best_f1 = -1.0
    best_threshold = thresholds[0] if thresholds else 0.0

    for threshold in thresholds:
        if detector._is_higher_score_better():
            total_pred = [int(score >= threshold) for score in total_score]
        else:
            total_pred = [int(score <= threshold) for score in total_score]
        f1 = f1_score(total_label, total_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    detector.threshold = float(best_threshold)
    return {
        'threshold': detector.threshold,
        'f1': best_f1
    }

def eval_threshold_detector(detector: ThresholdDetector, data_loader: DataLoader):
    total_label, total_pred, total_score = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            id, entropy_mean, entropy_max, prob_mean, prob_max, label = batch
            score = detector.score(entropy_mean, entropy_max, prob_mean, prob_max)
            score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
            pred = detector.predict(entropy_mean, entropy_max, prob_mean, prob_max)

            total_label += label.cpu().tolist()
            total_pred += pred.cpu().reshape(-1).tolist()
            total_score += score.cpu().reshape(-1).tolist()

        acc = accuracy_score(total_label, total_pred)
        f1 = f1_score(total_label, total_pred, zero_division=0)
        pr_scores = total_score if detector._is_higher_score_better() else [-score for score in total_score]
        precision_curve, recall_curve, _ = precision_recall_curve(total_label, pr_scores)
        pr_auc = auc(recall_curve, precision_curve)

    return {
        'acc': acc,
        'f1': f1,
        'pr_auc': pr_auc
    }

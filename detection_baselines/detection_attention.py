import gc
import os
from dataclasses import dataclass

import torch
from sklearn.metrics import (accuracy_score, auc, f1_score,
                              precision_recall_curve, precision_score,
                              recall_score)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from egh_vlm.utils import Qwen3ModelBundle

# NOTE: Requires the model to be loaded with attn_implementation='eager'.
# FlashAttention2 does not materialise attention weights, so output_attentions=True
# will either raise or return None with flash_attn. Example load:
#   model = AutoModelForCausalLM.from_pretrained(..., attn_implementation='eager')


def _get_visual_token_mask(input_ids: torch.Tensor, processor) -> torch.Tensor:
    """
    Boolean mask of visual patch-token positions.
    Tokens between <|vision_start|> and <|vision_end|> are marked True.
    Returns all-False mask if special tokens are unavailable (e.g. text-only input).
    """
    try:
        start_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
        end_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    except Exception:
        return torch.zeros(input_ids.shape[-1], dtype=torch.bool)

    ids = input_ids.squeeze(0).tolist()
    mask = torch.zeros(len(ids), dtype=torch.bool)
    in_vision = False

    for i, tid in enumerate(ids):
        if tid == start_id:
            in_vision = True
        elif tid == end_id:
            in_vision = False
        elif in_vision:
            mask[i] = True

    return mask


@dataclass
class AttentionFeatures:
    # Both scores are oriented so that higher value = more likely hallucination.
    vtar: torch.Tensor   # 1 - visual_token_attention_ratio: low visual attention → hallucination
    sink: torch.Tensor   # attention-sink ratio on token-0: high sink → model ignores visual tokens


class AttentionDataset(Dataset):
    def __init__(self, ids=None, vtars=None, sinks=None, labels=None):
        self.ids = ids if ids is not None else []
        self.vtars = vtars if vtars is not None else []
        self.sinks = sinks if sinks is not None else []
        self.labels = labels if labels is not None else []

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        return self.ids[idx], self.vtars[idx], self.sinks[idx], self.labels[idx]

    def get_by_id(self, id: str):
        try:
            return self.__getitem__(self.ids.index(id))
        except ValueError:
            return None

    def add_item(self, id: str, vtar: torch.Tensor, sink: torch.Tensor, label: int):
        self.ids.append(id)
        self.vtars.append(vtar.squeeze())
        self.sinks.append(sink.squeeze())
        self.labels.append(label)


def attention_collate_fn(batch):
    ids, vtars, sinks, labels = [], [], [], []
    for item in batch:
        ids.append(item[0])
        vtars.append(item[1])
        sinks.append(item[2])
        labels.append(item[3])
    return ids, torch.stack(vtars), torch.stack(sinks), torch.tensor(labels)


def save_attention_dataset(dataset, path):
    torch.save({
        'ids': dataset.ids,
        'vtars': dataset.vtars,
        'sinks': dataset.sinks,
        'labels': dataset.labels
    }, path)


def load_attention_dataset(path):
    ckpt = torch.load(path, map_location='cpu')
    return AttentionDataset(ckpt['ids'], ckpt['vtars'], ckpt['sinks'], ckpt['labels'])


def extract_attention_qwen3(
    messages: list,
    answer_messages: list,
    model_bundle: Qwen3ModelBundle,
    targeted_layer: int = -1) -> AttentionFeatures:
    """
    Single forward pass with output_attentions=True on the full (context + answer) sequence.

    VTAR (inverted): 1 - mean(answer_attn → visual_tokens) / mean(answer_attn → all_tokens)
    Sink: mean(answer_attn → token_0) / mean(answer_attn → all_tokens)

    Only the targeted_layer attention is used; heads are mean-pooled.
    """
    model, processor, device = model_bundle.model, model_bundle.processor, model_bundle.device

    ids = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors='pt'
    )
    a_ids = processor.apply_chat_template(
        answer_messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors='pt'
    )
    ids = {k: v.to(device) for k, v in ids.items()}
    a_length = a_ids['input_ids'].shape[1]
    seq_len = ids['input_ids'].shape[1]

    with torch.no_grad():
        model.eval()
        output = model(**ids, output_attentions=True)

    if output.attentions is None:
        raise RuntimeError(
            "output.attentions is None. Load the model with attn_implementation='eager'."
        )

    # (n_heads, seq_len, seq_len) → mean over heads → (seq_len, seq_len)
    attn = output.attentions[targeted_layer].squeeze(0).mean(dim=0).float()

    visual_mask = _get_visual_token_mask(ids['input_ids'], processor)  # (seq_len,)
    answer_start = seq_len - a_length + 1  # shift by 1: logit at t predicts token t+1

    if answer_start >= seq_len or answer_start < 0:
        vtar = torch.tensor([0.5], dtype=torch.float32)
        sink = torch.tensor([0.0], dtype=torch.float32)
    else:
        answer_attn = attn[answer_start:, :]  # (answer_len, seq_len)
        total_mean = answer_attn.mean().item()

        raw_vtar = (
            answer_attn[:, visual_mask].mean().item() / (total_mean + 1e-8)
            if visual_mask.any() else 0.0
        )
        vtar = torch.tensor([1.0 - raw_vtar], dtype=torch.float32)

        sink_val = answer_attn[:, 0].mean().item() / (total_mean + 1e-8)
        sink = torch.tensor([sink_val], dtype=torch.float32)

    vtar, sink = vtar.to('cpu'), sink.to('cpu')
    del ids, output, attn, visual_mask
    return AttentionFeatures(vtar=vtar, sink=sink)


def extract_attention(
    dataset,
    model_bundle,
    client_type: str = 'qwen3',
    save_path: str = None,
    save_interval: int = 20,
    mask_mode: str = None,
    targeted_layer: int = -1):
    """
    client_type    : 'qwen3'
    mask_mode      : None | 'image' | 'question'
    targeted_layer : which transformer layer to read attention from

    Requires model loaded with attn_implementation='eager'.
    """
    if client_type not in ['qwen3']:
        print('Unsupported client')
        return None

    if mask_mode not in [None, 'image', 'question']:
        print('Incorrect mask mode')
        return None

    processed = AttentionDataset()
    if save_path is not None and os.path.exists(save_path):
        processed = load_attention_dataset(save_path)
    processed_ids = set(processed.ids)

    for item in tqdm(dataset, desc=f'Extracting attention features ({client_type})'):
        if item['id'] in processed_ids:
            continue

        context = []
        if item['image_path'] is not None and mask_mode != 'image':
            context.append({'type': 'image', 'image': item['image_path']})
        if item['question'] is not None and mask_mode != 'question':
            context.append({'type': 'text', 'text': item['question']})

        messages = [
            {'role': 'user', 'content': context},
            {'role': 'assistant', 'content': [{'type': 'text', 'text': item['answer']}]}
        ]
        answer_messages = [
            {'role': 'assistant', 'content': [{'type': 'text', 'text': item['answer']}]}
        ]

        features = extract_attention_qwen3(
            messages=messages,
            answer_messages=answer_messages,
            model_bundle=model_bundle,
            targeted_layer=targeted_layer
        )
        vtar, sink = features.vtar, features.sink

        def _valid(t): return t.numel() > 0 and not torch.isnan(t).any() and not torch.isinf(t).any()
        if not (_valid(vtar) and _valid(sink)):
            print(f"Skipping id={item['id']}: invalid features.")
            continue

        processed.add_item(item['id'], vtar, sink, item['label'])

        if save_path is not None and len(processed) % save_interval == 0:
            save_attention_dataset(processed, save_path)

        del features, vtar, sink, messages, answer_messages, context
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if save_path is not None:
        save_attention_dataset(processed, save_path)
    return processed


class AttentionThresholdDetector:
    def __init__(self, metric_type: str = 'vtar', threshold: float = None):
        if metric_type not in ['vtar', 'sink']:
            raise ValueError("metric_type must be 'vtar' or 'sink'")
        self.metric_type = metric_type
        self.threshold = threshold

    def score(self, vtar: torch.Tensor, sink: torch.Tensor) -> torch.Tensor:
        return (vtar if self.metric_type == 'vtar' else sink).float()

    def predict(self, vtar: torch.Tensor, sink: torch.Tensor) -> torch.Tensor:
        if self.threshold is None:
            raise ValueError('Detector threshold is not trained.')
        return self.score(vtar, sink) >= self.threshold


def train_attention_detector(detector: AttentionThresholdDetector, data_loader: DataLoader):
    total_label, total_score = [], []

    for _, vtar, sink, label in data_loader:
        score = torch.nan_to_num(detector.score(vtar, sink), nan=0.0, posinf=1.0, neginf=0.0)
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


def eval_attention_detector(detector: AttentionThresholdDetector, data_loader: DataLoader):
    total_label, total_pred, total_score = [], [], []

    for _, vtar, sink, label in data_loader:
        score = torch.nan_to_num(detector.score(vtar, sink), nan=0.0, posinf=1.0, neginf=0.0).squeeze(-1)
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
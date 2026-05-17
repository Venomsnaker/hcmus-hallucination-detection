import gc
import os
from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    auc
)

from egh_vlm.utils import Qwen3ModelBundle


@dataclass
class HALPFeatures:
    vf: torch.Tensor # per-layer mean pooled output from the vision encoder
    vt: torch.Tensor # per-layer hidden states at the final position of the visual token sequence
    qt: torch.Tensor # per-layer hidden states at the final position of the query token sequence
    
class HALPDataset(Dataset):
    def __init__(self, ids=[], vfs=[], vts=[], qts=[], labels=[]):
        self.ids = ids
        self.vfs = vfs
        self.vts = vts
        self.qts = qts
        self.labels = labels
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx: int):
        return self.ids[idx], self.vfs[idx], self.vts[idx], self.qts[idx], self.labels[idx]
    
    def get_by_id(self, id: str):
        try:
            return self.__getitem__(self.ids.index(id))
        except ValueError:
            return None
        
    def add_item(self, id: str, features: HALPFeatures, label: int):
        self.ids.append(id)
        self.vfs.append(features.vf)
        self.vts.append(features.vt)
        self.qts.append(features.qt)
        self.labels.append(label)
        
def halp_collate_fn(batch):
    ids, vfs, vts, qts, labels = [], [], [], [], []
    
    for item in batch:
        id, vf, vt, qt, label = item
        ids.append(id)
        vfs.append(vf)
        vts.append(vt)
        qts.append(qt)
        labels.append(label)
    
    return ids, torch.stack(vfs), torch.stack(vts), torch.stack(qts), torch.tensor(labels)

def save_halp_dataset(dataset: HALPDataset, save_path: str):
    torch.save({
        'ids': dataset.ids,
        'vfs': dataset.vfs,
        'vts': dataset.vts,
        'qts': dataset.qts,
        'labels': dataset.labels
    }, save_path)
    
def load_halp_dataset(save_path: str) -> HALPDataset:
    checkpoint = torch.load(save_path, map_location='cpu')
    return HALPDataset(
        ids=checkpoint['ids'],
        vfs=checkpoint['vfs'],
        vts=checkpoint['vts'],
        qts=checkpoint['qts'],
        labels=checkpoint['labels']
    )
    
def extract_halp_features_qwen3(
    messages: list,
    model_bundle: Qwen3ModelBundle,
    target_layer: int = 21, # 3/4 depth
) -> HALPFeatures:
    model, processor, device = model_bundle.model, model_bundle.processor, model_bundle.device
    layer_module = model.model.language_model.layers
    
    ids = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors='pt')
    ids = {k: v.to(device) for k,v in ids.items()}
    
    input_ids = ids['input_ids'][0]
    attention_mask = ids.get('attention_mask', None)
    vision_start_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')
    
    vs_pos = (input_ids == vision_start_id).nonzero(as_tuple=True)[0][0].item()
    ve_pos = (input_ids == vision_end_id).nonzero(as_tuple=True)[0][0].item()
    vt_pos = ve_pos - 1
    
    if attention_mask is not None:
        qt_pos = attention_mask[0].nonzero(as_tuple=True)[0][-1].item()
    else:
        qt_pos = input_ids.shape[0] - 1
    
    buf = {'vf': None, 'vt': None, 'qt': None}

    pixel_values = ids.get('pixel_values', None)
    image_grid_thw = ids.get('image_grid_thw', None)
    if pixel_values is None or image_grid_thw is None:
        raise ValueError('Missing vision inputs for HALP VF extraction (pixel_values or image_grid_thw).')

    with torch.no_grad():
        model.eval()
        image_outputs = model.get_image_features(
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        pooled = image_outputs.pooler_output
        pooled = pooled[0] if isinstance(pooled, (list, tuple)) else pooled
        buf['vf'] = pooled.mean(dim=0).detach().cpu()
    
    def layer_hook(module, input, output):
        hidden_states = output[0]
        if hidden_states.dim() == 3:
            buf['vt'] = hidden_states[0, vt_pos].detach().cpu()
            buf['qt'] = hidden_states[0, qt_pos].detach().cpu()
        elif hidden_states.dim() == 2:
            buf['vt'] = hidden_states[vt_pos].detach().cpu()
            buf['qt'] = hidden_states[qt_pos].detach().cpu()
        else:
            raise ValueError(f'Unexpected hidden state shape for VT/QT: {tuple(hidden_states.shape)}')
        
    layer_idx = target_layer - 1 if target_layer > 0 else target_layer
    l_handle = layer_module[layer_idx].register_forward_hook(layer_hook)
    
    with torch.no_grad():
        model.eval()
        model(**ids)
    l_handle.remove()
    
    result = HALPFeatures(
        vf=buf['vf'],
        vt=buf['vt'],
        qt=buf['qt']
    )
    del ids, buf
    return result
    
def extract_halp(
    dataset,
    model_bundle: Qwen3ModelBundle,
    client_type: str='qwen3',
    save_path: str=None,
    save_interval: int=20,
    targeted_layer: int=2
) -> HALPDataset:
    """
    client_type: 'qwen3'
    """
    if client_type not in ['qwen3']:
        print('Unsupported client')
        return None

    processed_features = HALPDataset()
    
    if save_path is not None and os.path.exists(save_path):
        processed_features = load_halp_dataset(save_path)
    processed_ids = set(processed_features.ids)
    
    if client_type == 'qwen3':
        for item in tqdm(dataset, desc=f'Extracting HALP features for client {client_type}'):
            if item['id'] in processed_ids:
                continue
            
            image_path = item['image_path']
            question = item['question']
            answer = item['answer']
            
            # Construct messages context
            context = []
            
            if image_path is not None:
                context.append({'type': 'image', 'image': image_path})
            if question is not None:
                context.append({'type': 'text', 'text': question})
            
            messages = [
                {'role': 'user', 'content': context},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}
            ]
            
            # Extract features
            features = extract_halp_features_qwen3(
                messages=messages,
                model_bundle=model_bundle,
                target_layer=targeted_layer,
            )
            
            valid = all(
                t.numel() > 0 and not torch.isnan(t).any() and not torch.isinf(t).any()
                for t in [features.vf, features.vt, features.qt]
            )
            if not valid:
                print(f"Skipping id={item['id']}: invalid features.")
                continue
            
            processed_features.add_item(id=item['id'], features=features, label=item['label'])
            
            if save_path is not None and len(processed_features) % save_interval == 0:
                save_halp_dataset(processed_features, save_path)
            
            del features, context
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        if save_path is not None:
            save_halp_dataset(processed_features, save_path)
    return processed_features

class HALPMLP(nn.Module):
    def __init__(self, 
        input_dim: int, 
        device: torch.device, 
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512, dtype=dtype),
            nn.ReLU(),
            nn.Linear(512, 256, dtype=dtype),
            nn.ReLU(),
            nn.Linear(256, 128, dtype=dtype),
            nn.ReLU(),
            nn.Linear(128, 1, dtype=dtype),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype)
        return torch.sigmoid(self.net(x))

def _select_feature(features: HALPFeatures, feature_key: str) -> torch.Tensor:
    if feature_key == 'vf':
        return features.vf
    if feature_key == 'vt':
        return features.vt
    if feature_key == 'qt':
        return features.qt
    raise ValueError(f"Unknown feature_key: {feature_key!r}. Use 'vf', 'vt', or 'qt'.")

def train_halp_detector(
    dataset: HALPDataset,
    feature_key: str,
    input_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 32,
    seed: int = 42,
    train_ratio: float = 0.8,
    epoch_count: int = 50,
    lr: float = 1e-3,
):
    train_set, val_set = split_halp_dataset(dataset, train_ratio=train_ratio, seed=seed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=halp_collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=halp_collate_fn)

    detector = HALPMLP(input_dim=input_dim, device=device, dtype=dtype)
    optim = torch.optim.Adam(detector.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    for _ in range(epoch_count):
        detector.train()
        for _, vfs, vts, qts, labels in train_loader:
            if feature_key == 'vf':
                feats = vfs
            elif feature_key == 'vt':
                feats = vts
            elif feature_key == 'qt':
                feats = qts
            else:
                raise ValueError(f"Unknown feature_key: {feature_key!r}. Use 'vf', 'vt', or 'qt'.")

            optim.zero_grad()
            output = detector(feats).squeeze(-1)
            label = labels.to(device=output.device, dtype=output.dtype)
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0).clamp(1e-6, 1.0 - 1e-6)
            loss = loss_fn(output, label.clamp(0.0, 1.0))
            loss.backward()
            optim.step()

    metrics = eval_halp_detector(detector, val_loader, feature_key)
    return detector, metrics


def eval_halp_detector(
    detector: HALPMLP,
    data_loader: DataLoader,
    feature_key: str,
):
    total_label, total_pred, total_score = [], [], []
    detector.eval()

    with torch.no_grad():
        for _, vfs, vts, qts, labels in data_loader:
            if feature_key == 'vf':
                feats = vfs
            elif feature_key == 'vt':
                feats = vts
            elif feature_key == 'qt':
                feats = qts
            else:
                raise ValueError(f"Unknown feature_key: {feature_key!r}. Use 'vf', 'vt', or 'qt'.")

            output = detector(feats).squeeze(-1)
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
            total_label += labels.tolist()
            total_pred += [round(x) for x in output.cpu().tolist()]
            total_score += output.cpu().tolist()

    precision, recall, _ = precision_recall_curve(total_label, total_score)
    return {
        'acc': accuracy_score(total_label, total_pred),
        'f1': f1_score(total_label, total_pred, zero_division=0),
        'pr_auc': auc(recall, precision),
    }

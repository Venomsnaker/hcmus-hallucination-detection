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
    target_layer: int = 21 # 3/4 depth
) -> HALPFeatures:
    model, processor, device = model_bundle.model, model_bundle.processor, model_bundle.device
    visual_module = model.model.visual
    layer_module = model.language_model.layers
    
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
        qt_pos = attention_mask[0].sum().item() - 1
    else:
        qt_pos = input_ids.shape[0] - 1
    
    buf = {'vf': None, 'vt': None, 'qt': None}

    pixel_values = ids.get('pixel_values', None)
    image_grid_thw = ids.get('image_grid_thw', None)
    if pixel_values is None or image_grid_thw is None:
        raise ValueError('Missing vision inputs for HALP VF extraction (pixel_values or image_grid_thw).')

    with torch.no_grad():
        model.eval()
        visual_outputs = visual_module(
            pixel_values=pixel_values,
            grid_thw=image_grid_thw,
        )
        if hasattr(visual_outputs, 'last_hidden_state'):
            visual_tokens = visual_outputs.last_hidden_state
        elif isinstance(visual_outputs, (tuple, list)):
            visual_tokens = visual_outputs[0]
        else:
            visual_tokens = visual_outputs

        if visual_tokens.dim() == 3:
            buf['vf'] = visual_tokens[0].mean(dim=0).detach().cpu()
        elif visual_tokens.dim() == 2:
            buf['vf'] = visual_tokens.mean(dim=0).detach().cpu()
        else:
            raise ValueError(f'Unexpected visual output shape for VF extraction: {tuple(visual_tokens.shape)}')
    
    def layer_hook(module, input, output):
        hidden_states = output[0]
        buf['vt'] = hidden_states[0, vt_pos].detach().cpu()
        buf['qt'] = hidden_states[0, qt_pos].detach().cpu()
        
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
    

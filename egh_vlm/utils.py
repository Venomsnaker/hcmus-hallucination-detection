import os
import json
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split


@dataclass
class Qwen3ModelBundle:
    model: any
    processor: any
    device: torch.device
    
def get_response_qwen3(
    messages, 
    model_bundle: Qwen3ModelBundle, 
    max_new_tokens: int=64):
    model, processor, device = model_bundle.model, model_bundle.processor, model_bundle.device
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt'
    ).to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0] if len(output_text) == 1 else output_text
    
def save_dataset(dataset: list, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)

def get_pred(response):
    if 'yes' in response.strip()[:10].lower():
        return 1
    if 'no' in response.strip()[:10].lower():
        return 0
    return 0.5

def get_img_path(img_folder_path: str, img_name, dataset='phd') -> str:
    if dataset == 'phd':
        for subfolder_name in ['train2014', 'val2014']:
            subfolder_path = os.path.join(img_folder_path, subfolder_name)
            if os.path.exists(subfolder_path):
                local_img_name = f'COCO_{subfolder_name}_{img_name}.jpg'
                img_path = os.path.join(subfolder_path, local_img_name)
                if os.path.exists(img_path):
                    return img_path 
        print(f'Image {img_name} not found in PHD dataset.')
        return ''
    else:
        print('Dataset not recognized.')
        return ''

def load_phd_dataset(dataset_path: str, img_folder_path: str, sample_size: int=None) -> list:
    dataset = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_dataset = json.load(f)
    if sample_size is not None and len(raw_dataset) > sample_size:
        raw_dataset = raw_dataset[:sample_size]

    for item in raw_dataset:
        dataset.append({
            'id': item['id'],
            'couple_id': item['couple_id'],
            'task': item['task'],
            'hitem': item['hitem'],
            'subject': item['subject'],
            'gt': item['gt'],
            'question': item['question'],
            'image_path': get_img_path(img_folder_path, item['image_id'], 'phd'),
            'question_gt': item['question_gt'],
            'answer': item['answer'],
            'label': item['hallucinated_label'],
        })
    print(f'Successfully load the PhD dataset with: {len(dataset)} samples.')
    return dataset

def split_stratified(dataset, train_ratio=0.7, random_state=42):
    labels = np.array(dataset.labels)

    train_idx, test_idx = train_test_split(
        range(len(dataset)),
        test_size=1-train_ratio,
        stratify=labels,
        random_state=random_state
    )
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    return train_dataset, test_dataset

def create_balanced_dataloader(
    dataset,
    batch_size: int = 32,
    collate_fn=None,
    seed: int = 42,
):
    if isinstance(dataset, torch.utils.data.Subset):
        labels = np.array(dataset.dataset.labels)[dataset.indices]
    else:
        labels = np.array(dataset.labels)

    class_counts = np.bincount(labels.astype(int))
    class_weights = np.zeros_like(class_counts, dtype=np.float64)
    non_empty_classes = class_counts > 0
    class_weights[non_empty_classes] = 1.0 / class_counts[non_empty_classes]
    sample_weights = class_weights[labels.astype(int)]

    generator = torch.Generator()
    generator.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
    )

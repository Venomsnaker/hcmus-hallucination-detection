import os
import json
from dataclasses import dataclass
import torch


@dataclass
class ModelBundle:
    model: any
    processor: any
    device: torch.device

def get_pred(response):
    if 'yes' in response.strip()[:10].lower():
        return 1
    if 'no' in response.strip()[:10].lower():
        return 0
    return 0.5

def get_img_path(img_folder_path: str, img_name, dataset='phd') -> str:
    '''
    dataset: 'egh_vlm', 'phd', 'hallusion_bench', or 'halo_quest'
    '''
    if dataset == 'egh_vlm':
        return os.path.join(img_folder_path, img_name)
    elif dataset == 'phd':
        for subfolder_name in ['train2014', 'val2014']:
            subfolder_path = os.path.join(img_folder_path, subfolder_name)
            if os.path.exists(subfolder_path):
                local_img_name = f'COCO_{subfolder_name}_{img_name}.jpg'
                img_path = os.path.join(subfolder_path, local_img_name)
                if os.path.exists(img_path):
                    return img_path 
        print(f'Image {img_name} not found in PHD dataset.')
        return ''
    elif dataset == 'hallusion_bench':
        return img_folder_path + img_name[1:]
    elif dataset == 'halo_quest': 
        return ""
    else:
        print('Dataset not recognized.')
        return ''

def load_egh_dataset(dataset_path, img_folder_path, sample_size=None) -> list:
    dataset = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    if sample_size is not None and len(raw_data) > sample_size:
        raw_data = raw_data[:sample_size]

    for data in raw_data:
        data['image_path'] = get_img_path(img_folder_path, data['image_id'], 'egh_vlm')
        dataset.append(data)
    print(f'Successfully load the EHG dataset with: {len(dataset)} samples.')
    return dataset

def load_hallusion_bench_dataset(dataset_path, img_folder_path, sample_size=None) -> list:
    dataset = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    if sample_size is not None and len(raw_data) > sample_size:
        raw_data = raw_data[:sample_size]

    for data in raw_data:
        dataset.append({
            'id': data['id'],
            'category': data['category'],
            'subcategory': data['subcategory'],
            'question': data['question'],
            'image_path': get_img_path(img_folder_path, data['filename'], 'hallusion_bench'),
            'gt_answer': data['gt_answer'],
            'gt_answer_label': int(data['gt_answer_label']),
            'answer': data['qwen3_vl_2b_response'],
            'label': data['hallucinated_label'],
        })
    print(f'Successfully load the Hallusion Bench dataset with: {len(dataset)} samples.')
    return dataset

def load_phd_dataset(dataset_path, img_folder_path, sample_size=None) -> list:
    dataset = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    if sample_size is not None and len(raw_data) > sample_size:
        raw_data = raw_data[:sample_size]

    for data in raw_data:
        dataset.append({
            'id': data['id'],
            'couple_idx': data['couple_idx'],
            'task': data['task'],
            'hitem': data['hitem'],
            'subject': data['subject'],
            'gt': data['gt'],
            'question': data['question'],
            'image_path': get_img_path(img_folder_path, data['image_id'], 'phd'),
            'question_gt': data['question_gt'],
            'answer': data['qwen3_vl_2b_response'],
            'label': data['hallucinated_label'],
        })
    print(f'Successfully load the PhD dataset with: {len(dataset)} samples.')
    return dataset
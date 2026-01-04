import os
import json
from dataclasses import dataclass
import torch


@dataclass
class ModelBundle:
    model: any
    processor: any
    device: torch.device

def get_verdict(response):
    if 'yes' in response.strip()[:10].lower():
        return 1
    else:
        return 0

def load_egh_dataset(dir_path, file_name='egh_vlm.json', imgs_dir_name='images/', sample_size=None) -> list:
    dataset_path = os.path.join(dir_path, file_name)
    imgs_dir_path = os.path.join(dir_path, imgs_dir_name)
    dataset = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    if sample_size is not None and len(data_list) > sample_size:
        data_list = data_list[:sample_size]

    for data in data_list:
        data['image_path'] = imgs_dir_path + data['image_id']
        dataset.append(data)
    print(f"Successfully load the EHG dataset with: {len(dataset)} samples.")
    return dataset

def load_hallusion_bench_dataset(dir_path, file_name='hallusion_bench.json', imgs_dir_name='images/', sample_size=None) -> list:
    dataset_path = os.path.join(dir_path, file_name)
    images_path = os.path.join(dir_path, imgs_dir_name)
    dataset = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    if sample_size is not None and len(data_list) > sample_size:
        data_list = data_list[:sample_size]

    for data in data_list:
        dataset.append({
            "id": data['id'],
            "question": data['question'],
            'answer': data['qwenvl_answer'],
            "image_path": images_path + data['filename'][2:],
            "category": data['category'],
            "subcategory": data['subcategory'],
            "gt_answer": int(data['gt_answer']),
            "gt_answer_details": data['gt_answer_details'],
            "label": data['hallucination'],
        })
    print(f"Successfully load the Hallusion Bench dataset with: {len(dataset)} samples.")
    return dataset


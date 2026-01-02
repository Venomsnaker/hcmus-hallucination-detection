import os
import json
from datasets import load_dataset


def map_response(response):
    if 'yes' in response.lower():
        return 1
    else:
        return 0

def load_egh_dataset(dir_path, sample_size=None) -> list:
    dataset_path = dir_path + '/dataset.json'
    images_path = dir_path + '/images/'
    dataset = []

    with open(dataset_path, 'r') as f:
        data_list = json.load(f)
    if sample_size is not None and len(data_list) > sample_size:
        data_list = data_list[:sample_size]

    for data in data_list:
        dataset.append({
            "id": data['id'],
            "image_path": images_path + data['image_id'],
            "question": data['query'],
            "answer": data['answer'],
            "label": data['label']
        })
    print(f"Successfully load the EHG dataset with: {len(dataset)} samples.")
    return dataset

def load_hallusion_bench_dataset(dir_path, sample_size=None) -> list:
    dataset_path = dir_path + '/hallusion_bench.json'
    images_path = dir_path + '/images/'
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

def load_pope_dataset(split="test", sample_size=None) -> dict:
    pope_dataset = load_dataset('lmms-lab/POPE', split=split)
    dataset = {}

    if sample_size is not None and len(pope_dataset) > sample_size:
        pope_dataset = pope_dataset.select(range(sample_size))

    for sample in pope_dataset:
        dataset[sample['id']] = {
            'question_id': sample['question_id'],
            'question': sample['question'],
            'ground_truth': map_response(sample['answer']),
            'image': sample['image'],
            'image_source': sample['image_source'],
            'category': sample['category'],
        }
    print(f"Successfully load the POPE dataset with: {len(dataset)} samples.")
    return dataset


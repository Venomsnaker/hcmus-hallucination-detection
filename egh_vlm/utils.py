import os
import json
from datasets import load_dataset


def map_response(response):
    if 'yes' in response.lower().strip():
        return 1
    else:
        return 0
    
def save_dataset(filepath, dataset):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(dataset, f, indent=4)

def load_egh_dataset(folder_path) -> list:
    dataset_path = folder_path + '/dataset.json'
    images_path = folder_path + '/images/'
    dataset = []

    with open(dataset_path, 'r') as f:
        data_list = json.load(f)
    for data in data_list:
        dataset.append({
            "id": data['id'],
            "image_path": images_path + data['image_id'],
            "query": data['query'],
            "answer": data['answer'],
            "label": data['label']
        })
    print(f"Successfully load the EHG dataset with: {len(dataset)} samples.")
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


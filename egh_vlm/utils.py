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
    if "yes" in response.strip()[:10].lower():
        return 1
    else:
        return 0
    
def save_dataset(dataset, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=4)

def get_img_path(img_folder_path: str, img_name, dataset="phd") -> str:
    """
    dataset: "phd" or "hallusion_bench"
    """
    if dataset == "phd":
        for subfolder_name in ["train2014", "val2014"]:
            subfolder_path = os.path.join(img_folder_path, subfolder_name)
            if os.path.exists(subfolder_path):
                local_img_name = f"COCO_{subfolder_name}_{img_name}.jpg"
                img_path = os.path.join(subfolder_path, local_img_name)
                if os.path.exists(img_path):
                    return img_path 
        print(f"Image {img_name} not found in PHD dataset.")
        return ""
    elif dataset == "hallusion_bench":
        return img_folder_path + img_name[1:]
    else:
        print("Dataset not recognized.")
        return ""

def load_egh_dataset(dataset_path, img_folder_path, sample_size=None) -> list:
    dataset = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    if sample_size is not None and len(data_list) > sample_size:
        data_list = data_list[:sample_size]

    for data in data_list:
        data["image_path"] = os.path.join(img_folder_path, data["image_id"])
        dataset.append(data)
    print(f"Successfully load the EHG dataset with: {len(dataset)} samples.")
    return dataset

def load_hallusion_bench_dataset(dataset_path, img_folder_path, sample_size=None) -> list:
    dataset = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    if sample_size is not None and len(data_list) > sample_size:
        data_list = data_list[:sample_size]

    for data in data_list:
        dataset.append({
            "id": data["id"],
            "question": data["question"],
            "answer": data["qwenvl_answer"],
            "image_path": get_img_path(img_folder_path, data["filename"], "hallusion_bench"),
            "category": data["category"],
            "subcategory": data["subcategory"],
            "gt_answer": int(data["gt_answer"]),
            "gt_answer_details": data["gt_answer_details"],
            "label": data["hallucination"],
        })
    print(f"Successfully load the Hallusion Bench dataset with: {len(dataset)} samples.")
    return dataset

def load_phd_dataset(dataset_path, img_folder_path, sample_size=None) -> list:
    dataset = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        data_list = json.load(f)
    if sample_size is not None and len(data_list) > sample_size:
        data_list = data_list[:sample_size]
    
    for data in data_list:
        dataset.append({
            "id": data["id"],
            "question": data["question"],
            "answer": data["qwen3_vl_2b_response"],
            "image_path": get_img_path(img_folder_path, data["image_id"], "phd"),
            "label": data["hallucinated_label"],
            "task": data["task"],
            "context": data["context"],
            "hitem": data["hitem"],
            "subject": data["subject"],
            "gt": data["gt"],
            "question_gt": data["label"]
        })
    print(f"Successfully load the PhD dataset with: {len(dataset)} samples.")
    return dataset
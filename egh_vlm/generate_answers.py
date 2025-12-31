import os
import json
from tqdm import tqdm

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from egh_vlm.utils import load_pope_dataset, map_response, save_dataset
from egh_vlm.model import get_response


def generate_answers(saved_filepath, sample_size, checkpoint_interval=20):
    # Set up the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        dtype="auto",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    # Set up the dataset
    dataset = load_pope_dataset(split='test', sample_size=sample_size)
    res = {}

    if os.path.exists(saved_filepath):
        with open(saved_filepath, 'r') as f:
            res = json.load(f)

    # Get responses
    batch = []

    for sample_id in tqdm(dataset.keys(), desc="Processing POPE samples:"):
        if sample_id in res:
            continue
        sample = dataset[sample_id]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample['image']},
                    {"type": "text", "text": sample['question'] + " Answer only 'yes' or 'no'."},
                ],
            }
        ]
        response = get_response(messages, model, processor, max_new_tokens=8)

        item = {
            'question_id': sample['question_id'],
            'question': sample['question'],
            'ground_truth': sample['ground_truth'],
            'answer': response,
            'label': map_response(response),
        }
        batch.append((sample_id, item))

        if len(batch) >= checkpoint_interval:
            for sid, item_data in batch:
                res[sid] = item_data
            save_dataset(saved_filepath, res)
            batch = []
    if batch:
        for sid, item_data in batch:
            res[sid] = item_data
        save_dataset(saved_filepath, res)

    print(f"Completed. The total number of samples processed: {len(res)}")

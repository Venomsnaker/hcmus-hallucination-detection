import os
import json
from tqdm import tqdm

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from egh_vlm.utils import map_response

def save_dataset(filepath, dataset):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)

def get_response(messages, model, processor, max_new_tokens=512):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0] if len(output_text) == 1 else output_text

def generate_answers(saved_filepath, sample_size=None, checkpoint_interval=20):
    # Set up
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        dtype="auto",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    dataset_path = "../data/hallusion_bench"
    images_path = "/images/"
    res = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    if sample_size is not None and len(data_list) > sample_size:
        data_list = data_list[:sample_size]

    if os.path.exists(saved_filepath):
        with open(saved_filepath, 'r', encoding='utf-8') as f:
            res = json.load(f)
    processed_ids = [data['id'] for data in res]

    # Get responses
    batch = []

    for sample in tqdm(data_list, desc="Processing Hallusion Bench samples:"):
        if sample["id"] in processed_ids:
            continue
        image_path = images_path + sample['filename'][2:]

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": sample['question'] + " Answer in this format: yes/no, explain"},
                ],
            }
        ]
        response = get_response(messages, model, processor, max_new_tokens=512)
        sample['qwenvl_answer'] = response
        sample['hallucination'] = map_response(response)
        batch.append(sample)

        if len(batch) >= checkpoint_interval:
            res.extend(batch)
            save_dataset(saved_filepath, res)
            batch = []
    if batch:
        res.extend(batch)
        save_dataset(saved_filepath, res)
    print(f"Completed. The total number of samples processed: {len(res)}")

import os
import json
from tqdm import tqdm
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from egh_vlm.utils import save_dataset, get_verdict


def get_response(messages, model, processor, max_new_tokens=512):
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt'
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0] if len(output_text) == 1 else output_text

def generate_answers(dir_path, save_path, file_name='hallusion_bench.json', imgs_dir_name='images/', sample_size=None, save_interval=20):
    # Load dataset
    dataset_path = os.path.join(dir_path, file_name)
    imgs_dir_path = os.path.join(dir_path, imgs_dir_name)
    res = []
    processed_ids = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    if sample_size is not None and len(dataset) > sample_size:
        dataset = dataset[:sample_size]
    
    if os.path.exist(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            res = json.load(f)
    processed_ids = [data['id'] for data in res]
    
    # Init model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        'Qwen/Qwen3-VL-2B-Instruct',
        dtype="auto",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen3-VL-2B-Instruct',
        max_pixels=1280 * 720    
    )

    # Get responses
    batch = []

    for data in tqdm(dataset, desc=f"Processing {file_name}:"):
        if data['id'] in processed_ids:
            continue
        
        img_path = imgs_dir_path + data['filename'][2:]
        messages = [
            { 
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": data['question'] + "\nAnswer in this format: yes/no, explain the answer."}],
            }
        ]
        response = get_response(messages, model, processor)
        data['qwenvl_answer'] = response
        data['hallucination'] = get_verdict(response)
        batch.append(data)

        if len(batch) > save_interval:
            dataset.extend(batch)
            save_dataset(dataset, save_path)
            batch.clear()
    if batch:
        dataset.extend(batch)
        save_dataset(dataset, save_path)
    print(f"Completed. Dataset size: {len(dataset)}")

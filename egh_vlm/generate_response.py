import os
import json
from tqdm import tqdm
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from egh_vlm.utils import get_img_path, get_pred


def save_dataset(dataset, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)

def get_response(messages, model, processor, max_new_tokens=64):
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

def generate_raw_responses(save_path, dataset_path, img_folder_path, prompt_path, sample_size=None, save_interval=20, ids_range=None):
    # Load dataset
    result = []
    processed_ids = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    if sample_size is not None and len(dataset) > sample_size:
        dataset = dataset[:sample_size]
    
    if ids_range is not None:
        dataset = [d for d in dataset if d['id'] >= ids_range[0] and d['id'] <= ids_range[1]]

    # Load processed dataset
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
    processed_ids = [data['id'] for data in result]

    # Load prompt
    with open(prompt_path, 'r', encoding='utf-8') as file:
        prompt = file.read()

    # Init model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        'Qwen/Qwen3-VL-2B-Instruct',
        dtype='auto',
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen3-VL-2B-Instruct',
        max_pixels=1280 * 720    
    )

    # Generate responses
    batch = []

    for data in tqdm(dataset, desc=f'Processing:'):
        if data['id'] in processed_ids:
            continue
        
        img_path = get_img_path(img_folder_path, data['image_id'], dataset='phd')
        question = prompt.format(question=data['question'])
        messages = [
            { 
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': img_path},
                    {'type': 'text', 'text': question}],
            }
        ]
        response = get_response(messages, model, processor)
        
        # Post-processing
        data['qwen3_vl_2b_response'] = response
        # Assign hallucination label based on yes/no in the response
        hallucinated_label = 0 if get_pred(response) == data['label'] else 1
        data['hallucinated_label'] = hallucinated_label
        batch.append(data)

        if len(batch) > save_interval:
            result.extend(batch)
            save_dataset(result, save_path)
            batch.clear()
    if batch:
        result.extend(batch)
        save_dataset(result, save_path)
    print(f'Completed. Dataset size: {len(result)}')
    return result

def generate_enhanced_responses_baseline(save_path, dataset_path, img_folder_path, prompt_path, sample_size=None, save_interval=20, ids_range=None):
    # Load dataset
    result = []
    processed_ids = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    if sample_size is not None and len(dataset) > sample_size:
        dataset = dataset[:sample_size]
    
    if ids_range is not None:
        dataset = [d for d in dataset if d['id'] >= ids_range[0] and d['id'] <= ids_range[1]]

    # Load processed dataset
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
    processed_ids = [data['id'] for data in result]

    # Load prompt
    with open(prompt_path, 'r', encoding='utf-8') as file:
        prompt = file.read()

    # Init model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        'Qwen/Qwen3-VL-2B-Instruct',
        dtype='auto',
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen3-VL-2B-Instruct',
        max_pixels=1280 * 720    
    )

    # Generate responses
    batch = []

    for data in tqdm(dataset, desc=f'Processing:'):
        if data['id'] in processed_ids:
            continue
        
        # Decide whether the response is hallucinated or not
        pred = round(data['detector_score'])
        response = ''

        if pred == 1:
            img_path = get_img_path(img_folder_path, data['image_id'], dataset='phd')
            question = prompt.format(question=data['question'], incorrect_answer=data['qwen3_vl_2b_response'])
            messages = [
                { 
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': img_path},
                        {'type': 'text', 'text': question}],
                }
            ]
            response = get_response(messages, model, processor)
        else:
            response = data['qwen3_vl_2b_response']
        
        # Post-processing
        data['enhanced_response'] = response
        data['enhanced_pred'] = get_pred(response)
        data['baseline_pred'] = get_pred(data['qwen3_vl_2b_response'])
        batch.append(data)

        if len(batch) > save_interval:
            result.extend(batch)
            save_dataset(result, save_path)
            batch.clear()
    if batch:
        result.extend(batch)
        save_dataset(result, save_path)
    print(f'Completed. Dataset size: {len(result)}')
    return result
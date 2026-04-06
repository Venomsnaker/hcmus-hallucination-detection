import os
import json
from tqdm import tqdm
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from egh_vlm.utils import ModelBundle, get_img_path, get_pred


def save_dataset(dataset: list, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)

def get_response(messages, model_bundle: ModelBundle, max_new_tokens: int=64):
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

def generate_answers(
        dataset_path: str, 
        img_folder_path: str,
        prompt_path: str,
        save_path: str,
        save_interval: int=20,
        dataset_name: str='phd',
        sample_size: int=None,
        specified_ids: list=None):
    '''
    dataset_name: 'egh_vlm', 'hallusion_bench' or 'phd'
    '''
    results = []
    processed_ids = []

    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    if sample_size is not None and len(dataset) > sample_size:
        dataset = dataset[:sample_size]
    
    # Limit dataset by specified_ids if provided
    if specified_ids is not None:
        dataset = [item for item in dataset if item['id'] in specified_ids]

    # Load processed results
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    processed_ids = [item['id'] for item in results]

    # Load prompt
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        'Qwen/Qwen3-VL-2B-Instruct',
        dtype='auto',
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen3-VL-2B-Instruct',
        max_pixels=1024*768  
    )
    model_bundle = ModelBundle(model=model, processor=processor, device=device)

    # Generate answers
    batch = []

    for item in tqdm(dataset, desc=f'Processing dataset {dataset_name}:'):
        if item['id'] in processed_ids:
            continue
        
        img_path = get_img_path(img_folder_path, item['image_id'], dataset=dataset_name)
        question = prompt.format(question=item['question'])
        messages = [
            { 
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': img_path},
                    {'type': 'text', 'text': question}],
            }
        ]
        answer = get_response(messages, model_bundle, 64)
        
        # Post-processing
        item['qwen3_vl_2b_response'] = answer
        # Assign hallucination label based on yes/no in the response
        pred = get_pred(answer)
        if pred == 0.5:
            print(f'Item ID {item["id"]} with response: "{answer}" can\'t be classified. Defaulting to non-hallucinated (0).')
            pred = 0
        hallucinated_label = 0 if get_pred(answer) == item['label'] else 1
        item['hallucinated_label'] = hallucinated_label
        batch.append(item)

        # Save intermediate results
        if len(batch) > save_interval:
            results.extend(batch)
            save_dataset(results, save_path)
            batch.clear()
    # Final save
    if batch:
        results.extend(batch)
        save_dataset(results, save_path)
    print(f'Completed. Dataset size: {len(results)}')
    return results

def generate_enhanced_answers(
    dataset_path: str, 
    img_folder_path: str,
    prompt_path: str,
    save_path: str,
    save_interval: int=20,
    dataset_name: str='phd',
    sample_size: int=None,
    specified_ids: list=None):
    '''
    dataset_name: 'egh_vlm', 'hallusion_bench' or 'phd'
    '''
    results = []
    processed_ids = []

    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    if sample_size is not None and len(dataset) > sample_size:
        dataset = dataset[:sample_size]
    
    # Limit dataset by specified_ids if provided
    if specified_ids is not None:
        dataset = [item for item in dataset if item['id'] in specified_ids]

    # Load processed results
    if os.path.exists(save_path):
        with open(save_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    processed_ids = [item['id'] for item in results]

    # Load prompt
    with open(prompt_path, 'r', encoding='utf-8') as f:
        prompt = f.read()

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        'Qwen/Qwen3-VL-2B-Instruct',
        dtype='auto',
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(
        'Qwen/Qwen3-VL-2B-Instruct',
        max_pixels=1024*768  
    )
    model_bundle = ModelBundle(model=model, processor=processor, device=device)

    # Generate answers
    batch = []

    for item in tqdm(dataset, desc=f'Processing:'):
        if item['id'] in processed_ids:
            continue
        
        # Predict whether the model response is hallucinated or not
        pred = round(item['detector_score'])

        # Is hallucinated, generate enhanced answer
        if pred == 1:
            img_path = get_img_path(img_folder_path, item['image_id'], dataset=dataset_name)
            question = prompt.format(question=item['question'], incorrect_answer=item['qwen3_vl_2b_response'])
            messages = [
                { 
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': img_path},
                        {'type': 'text', 'text': question}],
                }
            ]
            response = get_response(messages, model_bundle, 64)
        else:
            response = item['qwen3_vl_2b_response']
        
        # Post-processing
        item['enhanced_response'] = response
        item['enhanced_pred'] = get_pred(response)
        item['baseline_pred'] = get_pred(item['qwen3_vl_2b_response'])
        batch.append(item)
    
        # Save intermediate results
        if len(batch) > save_interval:
            results.extend(batch)
            save_dataset(results, save_path)
            batch.clear()
    # Final save
    if batch:
        results.extend(batch)
        save_dataset(results, save_path)
    print(f'Completed. Dataset size: {len(results)}')
    return results
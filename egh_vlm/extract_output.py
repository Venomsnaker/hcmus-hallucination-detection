from tqdm import tqdm
import gc
import torch

from egh_vlm.utils import ModelBundle

def extract_output_pipeline(model_bundle: ModelBundle, messages: list):
    model, processor, device = model_bundle.model, model_bundle.processor, model_bundle.device
    
    # Tokenize inputs
    ids = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors='pt'
    )
    ids = {k: v.to(device) for k, v in ids.items()}

    with torch.set_grad_enabled(True):
        model.eval()

        # Forward pass
        tensor_output = model(**ids, output_hidden_states=True)
    return tensor_output, ids['input_ids'].shape[1]

def extract_output(model_bundle: ModelBundle, answer: str, image_path: str = None, question: str = None, mask_mode=None):
    '''
    mask_mode: None, 'all', 'image', or 'question'
    '''
    if mask_mode not in [None, 'all', 'image', 'question']:
        print('Incorrect mask mode')
        return None

    context = []
    messages = []

    if image_path is not None and mask_mode not in ['image', 'all']:
        context.append({'type': 'image', 'image': image_path})
    if question is not None and mask_mode not in ['question', 'all']:
        context.append({'type': 'text', 'text': question})
    
    if context != []:
        messages.append({'role': 'user', 'content': context})
    messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]})
    
    output, tokenization_length = extract_output_pipeline(model_bundle, messages)
    return output, tokenization_length

def batch_extract_output(data_list, model_bundle: ModelBundle, processed_outputs: dict=None, mask_mode=None, save_path=None, save_interval=100):
    '''
    mask_mode: None, 'all', 'image' or 'question'
    '''
    if mask_mode not in [None, 'all', 'image', 'question']:
        print('Incorrect mask mode')
        return None
    
    if processed_outputs is None:
        processed_outputs = {}
    processed_ids = set(processed_outputs.keys())

    for data in tqdm(data_list, desc='Extract output:'):
        if data['id'] in processed_ids:
            continue

        tensor_output, tokenization_length = extract_output(
            model_bundle= model_bundle,
            answer=data['answer'],
            image_path=data['image_path'],
            question=data['question'],
            mask_mode=mask_mode
        )
        processed_outputs[data['id']] = {
            'hidden_states': tuple(h.detach().cpu() for h in tensor_output.hidden_states),
            'logits': tensor_output.logits.detach().cpu(),
            'tokenization_length': tokenization_length,
            'label': data['label']
        }
        
        if save_path is not None:
            if save_path is not None and len(processed_outputs) % save_interval == 0:
                torch.save(processed_outputs, save_path)

        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final save
    if save_path is not None:
        torch.save(processed_outputs, save_path)
    return processed_outputs
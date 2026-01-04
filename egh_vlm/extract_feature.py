from tqdm import tqdm
import gc
import torch

from egh_vlm.hallucination_dataset import HallucinationDataset, save_features
from egh_vlm.utils import ModelBundle


def extract_features_pipeline(model_bundle: ModelBundle, context_messages: list, answer_messages: list):
    model, processor, device = model_bundle.model, model_bundle.processor, model_bundle.device

    c_ids = processor.apply_chat_template(
        context_messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt"
    )
    a_ids = processor.apply_chat_template(
        answer_messages, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt"
    )
    c_ids = {k: v.to(device) for k, v in c_ids.items()}
    a_ids = {k: v.to(device) for k, v in a_ids.items()}

    with torch.set_grad_enabled(True):
        model.eval()

        c_output = model(**c_ids, output_hidden_states=True)
        a_output = model(**a_ids, output_hidden_states=True)

        c_length = c_ids["input_ids"].shape[1]
        a_length = a_ids["input_ids"].shape[1]

        # Extract answer prob (slice after context)
        c_prob = c_output.logits.squeeze(0)[c_length - a_length + 1:, :]
        a_prob = a_output.logits.squeeze(0)[1:, :]

        # Extract last hidden states
        c_vector = c_output.hidden_states[-1]
        a_vector = a_output.hidden_states[-1]

        # Compute KL divergence & gradient
        kl_divergence = torch.sum(
            a_prob.softmax(dim=-1) * (a_prob.softmax(dim=-1).log() - torch.log_softmax(c_prob, dim=-1))
        )
        grad = torch.autograd.grad(
            outputs=kl_divergence, inputs=a_vector, create_graph=False, allow_unused=True,
        )
        # Fallback to zeros if gradient is None
        if grad[0] is not None:
            grad = grad[0].squeeze(0)[1:, :]
        else:
            grad = torch.zeros_like(a_vector.squeeze(0)[1:, :])

        # Compute embedding
        a_emb = a_vector.squeeze(0)[1:, :]
        c_emb = c_vector.squeeze(0)[c_length - a_length + 1:, :]
        emb = c_emb - a_emb
    return emb.detach().float().to("cpu"), grad.detach().float().to("cpu")

def extract_features(model_bundle: ModelBundle, answer: str, image_path: str = None, question: str = None, mask_mode=None):
    '''
    mask_mode: None, 'image' or 'question'
    '''
    if mask_mode not in [None, 'image', 'question']:
        print('Incorrect mask mode')
        return None

    context = []

    if image_path is not None and mask_mode is not 'image':
        context.append({"type": "image", "image": image_path})
    if question is not None and mask_mode is not 'question':
        context.append({"type": "text", "text": question})
    
    context_messages = [
        {"role": "user", "content": context},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    ]
    answer_messages = [
        {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    ]

    emb, grad = extract_features_pipeline(model_bundle, context_messages, answer_messages)
    return emb, grad

def batch_extract_features(data_list, model_bundle: ModelBundle, mask_mode=None, save_path: str=None, save_interval=20):
    if mask_mode not in [None, 'image', 'question']:
        print('Incorrect mask mode')
        return None

    dataset = HallucinationDataset()

    for data in tqdm(data_list, desc='Extract features:'):
        emb, grad = extract_features(
            model_bundle,
            answer = data['answer'],
            image_path = data['image_path'],
            question=data['question'],
            mask_mode=mask_mode
        )

        # Exclude empty, NaN, and inf features 
        if emb.numel() > 0 and grad.numel() > 0:
            if not torch.isnan(emb).any() and not torch.isinf(emb).any() and not torch.isnan(grad).any() and not torch.isinf(grad).any():
                dataset.add_item(data['id'], emb, grad, data['label'])
        
        # Save features
        if save_path is not None:
            if len(dataset) % save_interval == 0:
                save_features(dataset, save_path)

        # Clean up
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return dataset

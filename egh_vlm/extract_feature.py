from tqdm import tqdm
import gc

import torch
from egh_vlm.hallucination_dataset import HallucinationDataset, save_features

def extract_features(
        question: str, image_path: str, answer: str, model, processor, device
):
    # question/image + answer
    q_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        }
    ]
    q_inputs = processor.apply_chat_template(
        q_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt"
    )
    q_inputs = {k: v.to(device) for k, v in q_inputs.items()}

    # answer
    a_messages = [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        }
    ]
    a_inputs = processor.apply_chat_template(
        a_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt"
    )
    a_inputs = {k: v.to(device) for k, v in a_inputs.items()}

    with torch.set_grad_enabled(True):
        model.eval()

        q_output = model(**q_inputs, output_hidden_states=True)
        a_output = model(**a_inputs, output_hidden_states=True)

        q_length = q_inputs["input_ids"].shape[1]
        a_length = a_inputs["input_ids"].shape[1]

        # Extract answer probs (slice after context)
        q_prob = q_output.logits.squeeze(0)[q_length - a_length + 1:, :]
        a_prob = a_output.logits.squeeze(0)[1:, :]

        # Extract last hidden states (embeddings)
        q_vector = q_output.hidden_states[-1]
        a_vector = a_output.hidden_states[-1]

        kl_divergence = torch.sum(
            a_prob.softmax(dim=-1) * (a_prob.softmax(dim=-1).log() - torch.log_softmax(q_prob, dim=-1))
        )
        gradient = torch.autograd.grad(
            outputs=kl_divergence, inputs=a_vector, create_graph=False, allow_unused=True,
        )
        if gradient[0] is not None:
            gradient = gradient[0].squeeze(0)[1:, :]
        else:
            gradient = torch.zeros_like(a_vector.squeeze(0)[1:, :])

        a_embedding = a_vector.squeeze(0)[1:, :]
        q_embedding = q_vector.squeeze(0)[q_length - a_length + 1:, :]
        embedding = q_embedding - a_embedding

    return (
        embedding.detach().float().to("cpu"),
        gradient.detach().float().to("cpu"))

def batch_extract_features(data_list, model, processor, device, saved_path=None):
    dataset = HallucinationDataset([], [], [])

    for data in tqdm(data_list, desc="Extract features:"):
        embedding, gradient = extract_features(
            question = data['question'],
            image_path= data['image_path'],
            answer = data['answer'],
            model=model,
            processor=processor,
            device=device
        )

        if embedding.numel() > 0 and gradient.numel() > 0:
            if not torch.isnan(embedding).any() and not torch.isinf(embedding).any():
                if not torch.isnan(gradient).any() and not torch.isinf(gradient).any():
                    dataset.embeddings.append(embedding)
                    dataset.gradients.append(gradient)
                    dataset.labels.append(data['label'])

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


        if saved_path is not None:
            if len(dataset) % 5 == 0:
                save_features(dataset, saved_path)
    return dataset


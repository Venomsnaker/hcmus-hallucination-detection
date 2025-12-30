import torch
from tqdm import tqdm

def extract_features(
        query: str, image_path: str, answer: str, model, processor, device
):
    # image + answer
    i_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        }
    ]
    i_inputs = processor.apply_chat_template(
        i_messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt"
    )
    i_inputs = {k: v.to(device) for k, v in i_inputs.items()}

    # query + answer
    q_messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
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
        i_output = model(**i_inputs, output_hidden_states=True)
        a_output = model(**a_inputs, output_hidden_states=True)

        q_length = q_inputs["input_ids"].shape[1]
        i_length = i_inputs["input_ids"].shape[1]
        a_length = a_inputs["input_ids"].shape[1]

        # extract the answer from the logits
        q_prob = q_output.logits.squeeze(0)[q_length - a_length + 1:, :]
        i_prob = i_output.logits.squeeze(0)[i_length - a_length + 1:, :]
        a_prob = a_output.logits.squeeze(0)[1:, :]

        # extract the embedding layer (last hidden states)
        q_vector = q_output.hidden_states[-1]
        i_vector = i_output.hidden_states[-1]
        a_vector = a_output.hidden_states[-1]

        a_vector_grad = a_vector.clone().detach().requires_grad_(True)
        a_embedding = a_vector.squeeze(0)[1:, :]

        # question - answer
        qa_kl_divergence = torch.sum(
            a_prob.softmax(dim=-1) *
            (a_prob.softmax(dim=-1).log() - torch.log_softmax(q_prob, dim=-1))
        )
        qa_gradient = torch.autograd.grad(
            outputs=qa_kl_divergence, inputs=a_prob, create_graph=False,
        )[0].squeeze(0)[1:, :]

        q_embedding = q_vector.squeeze(0)[q_length - a_length + 1:, :]
        qa_embedding = q_embedding - a_embedding

        # image - answer
        ia_kl_divergence = torch.sum(
            a_prob.softmax(dim=-1) *
            (a_prob.softmax(dim=-1).log() - torch.log_softmax(i_prob, dim=-1))
        )
        ia_gradient = torch.autograd.grad(
            outputs=ia_kl_divergence, inputs=a_prob, create_graph=False,
        )[0].squeeze(0)[1:, :]

        i_embedding = i_vector.squeeze(0)[i_length - a_length + 1:, :]
        ia_embedding = i_embedding - a_embedding
    return qa_embedding.detach().to("cpu"), qa_gradient.detach().to("cpu"), ia_embedding.detach().to("cpu"), ia_gradient.detach().to("cpu")

def batch_extract_features(data_list, model, processor, device):
    qa_embeddings = []
    qa_gradients = []
    ia_embeddings = []
    ia_gradients = []

    for data in tqdm(data_list):
        qa_embedding, qa_gradient, ia_embedding, ia_gradient = extract_features(
            query = data['query'],
            image_path= data['image'],
            answer = data['answer'],
            model=model,
            processor=processor,
            device=device
        )
        qa_embeddings.append(qa_embedding)
        qa_gradients.append(qa_gradient)
        ia_embeddings.append(ia_embedding)
        ia_gradients.append(ia_gradient)
    return qa_embeddings, qa_gradients, ia_embeddings, ia_gradients


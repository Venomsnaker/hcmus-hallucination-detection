import os
import json
from tqdm import tqdm
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from egh_vlm.utils import get_img_path, save_dataset, get_verdict


def get_response(messages, model, processor, max_new_tokens=64):
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

def generate_answers(save_path, folder_path, file_name, img_dir_name="images", sample_size=None, save_interval=20, ids_range=None):
    # Load dataset
    dataset_path = os.path.join(folder_path, file_name)
    img_folder_path = os.path.join(folder_path, img_dir_name)
    res = []
    processed_ids = []

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if sample_size is not None and len(dataset) > sample_size:
        dataset = dataset[:sample_size]
    
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            res = json.load(f)
    processed_ids = [data["id"] for data in res]

    if ids_range is not None:
        dataset = [d for d in dataset if d["id"] >= ids_range[0] and d["id"] <= ids_range[1]]

    # Init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        dtype="auto",
        device_map=device
    )
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct",
        max_pixels=1280 * 720    
    )

    # Get responses
    batch = []

    for data in tqdm(dataset, desc=f"Processing {file_name}:"):
        if data["id"] in processed_ids:
            continue
        
        img_path = get_img_path(img_folder_path, data["image_id"], dataset="phd")
        messages = [
            { 
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": data["question"] + "\nAnswer ONLY in this exact format, make sure you add the explanation: yes/no, explanation based on what you see in the image."}],
            }
        ]
        # For testing purpose
        # response = "yes, test answer."
        response = get_response(messages, model, processor)
        
        data["qwen3_vl_2b_response"] = response
        hallucinated_label = 0 if get_verdict(response) == data["label"] else 1
        data["hallucinated_label"] = hallucinated_label
        batch.append(data)

        if len(batch) > save_interval:
            res.extend(batch)
            save_dataset(res, save_path)
            batch.clear()
    if batch:
        res.extend(batch)
        save_dataset(res, save_path)
    print(f"Completed. Dataset size: {len(res)}")
    return res

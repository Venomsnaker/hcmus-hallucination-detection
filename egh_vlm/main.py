import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from egh_vlm.hallucination_detector import DetectorModule
from egh_vlm.extract_feature import batch_extract_features
from egh_vlm.hallucination_dataset import HallucinationDataset, hallucination_collate_fn
from egh_vlm.training import train, eval_detector
from egh_vlm.utils import load_egh_dataset

def load_dataset(dir_path, model, processor, device):
    data_list = load_egh_dataset(dir_path)
    qa_embedding, qa_gradient, ia_embedding, ia_gradient = (batch_extract_features(data_list, model, processor, device))
    labels = [data['label'] for data in data_list]
    hidden_size = qa_embedding[0].size(-1)
    return HallucinationDataset(
        qa_embedding,
        qa_gradient,
        ia_embedding,
        ia_gradient,
        labels,
    ), hidden_size

def run():
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir_path", type=str, default="data/egh_vlm")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--train_ratio", type=float, default=0.5)
    args, _ = parser.parse_known_args()

    # VLM model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_name)

    # Dataset
    dataset, hidden_size = load_dataset(args.dataset_dir_path, model, processor, device)
    train_dataset = dataset
    test_dataset = dataset
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        collate_fn=hallucination_collate_fn,
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        collate_fn=hallucination_collate_fn,
        shuffle=True,
    )

    # Training
    detector = DetectorModule(hidden_size, hidden_size, 1)
    epoch = 10
    loss_function = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(detector.parameters(), lr=1e-4)
    best_acc = 0.0
    best_f1 = 0.0

    for i in range(epoch):
        total_loss = train(loss_function, optim, train_dataloader)
        print(f'Epoch [{i + 1}/{epoch}], Loss: {total_loss / 2000:.4f}')
        acc, f1, pr_auc = eval_detector(detector, test_dataloader)
        print(f'Epoch [{i + 1}/{epoch}], ACC: {acc:.4f}, F1: {f1:.4f}, PR-AUC:{pr_auc:.4f}')

        if acc > best_acc:
            best_acc = acc
            print(f'Best ACC at epoch {epoch + 1}')
        if f1 > best_f1:
            best_f1 = f1
            print(f'Best F1 at epoch {epoch + 1}')

        if total_loss < 1e-5:
            break

    print(f'Training Finished!')

    # Eval detector



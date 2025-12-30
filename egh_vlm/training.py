from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, auc
import torch
from torch.utils.data import DataLoader, random_split
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from egh_vlm.hallucination_detector import DetectorModule
from egh_vlm.extract_feature import batch_extract_features
from egh_vlm.hallucination_dataset import HallucinationDataset, hallucination_collate_fn
from egh_vlm.utils import load_egh_dataset

def train(self, loss_function, optim, data_loader):
    



class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(self.args.device)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.args.model_name,
            dtype="auto",
            device_map=self.device
        )
        self.processor = AutoProcessor.from_pretrained(self.args.model_name)
        self.detector = DetectorModule(2048, 2048, 1)
        self.dataset = None
        self.load_dataset(self.args.dataset_dir_path)

    def load_dataset(self, path):
        data_list = load_egh_dataset(path)
        qa_embedding, qa_gradient, ia_embedding, ia_gradient = batch_extract_features(data_list, self.model, self.processor, self.device)
        labels = [data['label'] for data in data_list]
        self.dataset = HallucinationDataset(qa_embedding, qa_gradient, ia_embedding, ia_gradient, labels)
        return

    def train(self, loss_function, optimizer, data_loader):
        total_loss = 0

        for _, batch in enumerate(data_loader):
            optimizer.zero_grad()
            qa_embedding, qa_gradient, ia_embedding, ia_gradient, label = batch
            label = label.float()
            output = self.detector(qa_embedding, qa_gradient, ia_embedding, ia_gradient).squeeze()
            loss = loss_function(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss
        return total_loss

    def eval(self, data_loader):
        total_label, total_pred, total_out = [], [], []

        with torch.no_grad():
            for _, batch in enumerate(data_loader):
                qa_embedding, qa_gradient, ia_embedding, ia_gradient, label = batch

                output = self.detector(qa_embedding, qa_gradient, ia_embedding, ia_gradient).squeeze()
                total_out += output.tolist()
                total_label += label.tolist()
                pred = list(map(lambda x: round(x), output.tolist()))
                total_pred += pred
            f1 = f1_score(total_label, total_pred)
            acc = accuracy_score(total_label, total_pred)
            precision, recall, cm = precision_recall_curve(total_label, total_pred)
            pr_auc = auc(recall, precision)
        return acc, f1, pr_auc

    def run(self):
        if self.dataset:
            # Configure data loader
            train_size = int(self.args.train_ratio * len(self.dataset))
            test_size = int(len(self.dataset) - train_size)
            train_dataset, test_dataset = random_split(self.dataset, [train_size, test_size])
            train_data_loader = DataLoader(train_dataset, batch_size=4, collate_fn=hallucination_collate_fn, shuffle=True)
            test_data_loader = DataLoader(test_dataset, batch_size=4, collate_fn=hallucination_collate_fn, shuffle=True)

            # Training parameters
            epoch = 2
            lost_function = torch.nn.BCELoss()
            optim = torch.optim.Adam(self.detector.parameters(), lr=1e-3)

            for i in range(epoch):
                total_loss = self.train(lost_function, optim, train_data_loader)
                print(f'Epoch [{i + 1}/{epoch}], Loss: {total_loss / 2000:.4f}')
                acc, f1, pr_auc = self.eval(test_data_loader)
                print(f'Epoch [{i + 1}/{epoch}], ACC: {acc:.4f}, F1: {f1:.4f}, PR-AUC:{pr_auc:.4f}')

                if total_loss < 1:
                    break
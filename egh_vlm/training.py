from sklearn.metrics import f1_score, accuracy_score, precision_recall_curve, auc
import torch


def train(detector, loss_function, optimizer, data_loader):
    total_loss = 0

    for _, batch in enumerate(data_loader):
        optimizer.zero_grad()
        qa_embedding, qa_gradient, ia_embedding, ia_gradient, label = batch
        label = label.float()
        output = detector(qa_embedding, qa_gradient, ia_embedding, ia_gradient).squeeze()
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss
    return total_loss

def eval_detector(detector, data_loader):
    total_label, total_pred, total_out = [], [], []

    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            qa_embedding, qa_gradient, ia_embedding, ia_gradient, label = batch

            output = detector(qa_embedding, qa_gradient, ia_embedding, ia_gradient).squeeze()
            total_out += output.tolist()
            total_label += label.tolist()
            pred = list(map(lambda x: round(x), output.tolist()))
            total_pred += pred
        f1 = f1_score(total_label, total_pred, average='macro')
        acc = accuracy_score(total_label, total_pred)
        precision, recall, cm = precision_recall_curve(total_label, total_pred)
        pr_auc = auc(recall, precision)
    return acc, f1, pr_auc





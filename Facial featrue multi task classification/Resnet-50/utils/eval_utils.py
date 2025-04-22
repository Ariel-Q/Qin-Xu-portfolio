import torch
from sklearn.metrics import f1_score, accuracy_score

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            x = batch['image'].to(device)
            y = batch['label'].to(device)
            out = model(x)
            preds = (torch.sigmoid(out) > 0.5).float().cpu()
            all_preds.append(preds)
            all_labels.append(batch['label'])
    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    acc = (preds == labels).float().mean().item()
    f1 = f1_score(labels, preds, average='macro')
    print(f" Val Accuracy: {acc:.4f}, F1-macro: {f1:.4f}")
    return acc, f1

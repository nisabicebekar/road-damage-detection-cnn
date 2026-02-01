
import sys
import os
sys.path.append(os.path.abspath("."))

print("INFERENCE.PY ÇALIŞIYOR")


import torch
from torch.utils.data import DataLoader
from src.utils.rdd_dataset import RDDDataset
from src.models.cnn_model import SimpleCNN

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import numpy as np


BATCH_SIZE = 32
DEVICE = "cpu"
MODEL_PATH = "outputs/model.pth"


test_dataset = RDDDataset(
    image_dir="data/raw/RDD_SPLIT/val/images",
    label_dir="data/raw/RDD_SPLIT/val/labels"
)

test_dataset.images = test_dataset.images[:1000]

print(f"Test görüntü sayısı: {len(test_dataset)}")

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model yüklendi")


all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()

        all_preds.extend(preds.flatten())
        all_labels.extend(labels.numpy())


acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)

tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
iou = tp / (tp + fp + fn + 1e-8)


print("\n===== TEST SONUÇLARI =====")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"IoU       : {iou:.4f}")
print("==========================")
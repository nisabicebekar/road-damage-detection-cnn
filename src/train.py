
import sys
import os
sys.path.append(os.path.abspath("."))

print("TRAIN.PY ÇALIŞIYOR")


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.utils.rdd_dataset import RDDDataset
from src.models.cnn_model import SimpleCNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
DEVICE = "cpu"
MODEL_PATH = "outputs/model.pth"


train_dataset = RDDDataset(
    image_dir="data/raw/RDD_SPLIT/train/images",
    label_dir="data/raw/RDD_SPLIT/train/labels"
)

val_dataset = RDDDataset(
    image_dir="data/raw/RDD_SPLIT/val/images",
    label_dir="data/raw/RDD_SPLIT/val/labels"
)

train_dataset.images = train_dataset.images[:5000]
val_dataset.images = val_dataset.images[:1000]
print("Dataset ayarlandı → Train: 5000 | Val: 1000")


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


model = SimpleCNN().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


for epoch in range(EPOCHS):
    print(f"\nEPOCH {epoch + 1} BAŞLADI")
    model.train()
    train_loss = 0.0
    train_preds, train_labels = [], []

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
        train_preds.extend(preds.flatten())
        train_labels.extend(labels.cpu().numpy().flatten())

    train_acc = accuracy_score(train_labels, train_preds)


    model.eval()
    val_preds, val_labels = [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()

            val_preds.extend(preds.flatten())
            val_labels.extend(labels.numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    precision = precision_score(val_labels, val_preds, zero_division=0)
    recall = recall_score(val_labels, val_preds, zero_division=0)
    f1 = f1_score(val_labels, val_preds, zero_division=0)

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] | "
        f"Train Loss: {train_loss / len(train_loader):.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"P: {precision:.4f} | "
        f"R: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )


os.makedirs("outputs", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print("\nModel kaydedildi → outputs/model.pth ✅")
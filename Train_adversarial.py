import os
import html
import time
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

# HuggingFace tokenizer (recommended for clean CSV -> BERT inputs)
from transformers import BertTokenizerFast

from Model_PMA import Model  # your model


CSV_PATH = "Data/Grambedding_dataset_Adversarial_Full/train_aug_mode.csv"
MODEL_PATH = "model.pth"
CM_PATH = "confusion_matrix.png"


def load_csv_as_binary(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep="\t", header=None, names=["raw_label", "url"], engine="python")

    # Clean URL strings (remove leading/trailing quotes) + decode &amp; etc.
    df["url"] = df["url"].astype(str).str.strip()
    df["url"] = df["url"].str.strip('"').str.strip("'").str.strip()
    df["url"] = df["url"].apply(html.unescape)

    # ---- IMPORTANT: binary label rule ----
    df["label"] = (df["raw_label"].fillna(0).astype(np.int64) != 0).astype(np.int64)

    # Drop empty/invalid rows
    df = df[df["url"].str.len() > 0].reset_index(drop=True)
    return df[["url", "label"]]


def encode_urls(tokenizer, urls, max_len=128):
    enc = tokenizer(
        list(urls),
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"]
    token_type_ids = enc.get("token_type_ids", torch.zeros_like(input_ids))
    attention_mask = enc["attention_mask"]
    return input_ids, token_type_ids, attention_mask


def train_one_epoch(model, device, train_loader, optimizer, epoch):
    model.train()

    running_loss = 0.0
    for batch_idx, (input_ids, token_type_ids, attention_mask, y) in enumerate(train_loader):
        start_time = time.time()

        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        y = y.to(device).long().view(-1)  # (B,)

        optimizer.zero_grad(set_to_none=True)

        # Your model expects a list [x1, x2, x3]
        outputs, pooled, logits = model([input_ids, token_type_ids, attention_mask])

        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 100 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            print(
                f"Train Epoch: {epoch} [{(batch_idx+1)*len(y)}/{len(train_loader.dataset)} "
                f"({100.0*(batch_idx+1)/len(train_loader):.2f}%)]\t"
                f"Loss: {loss.item():.6f}\tAvgLoss: {avg_loss:.6f}"
            )


@torch.no_grad()
def validation(model, device, val_loader, cm_path=CM_PATH):
    model.eval()

    total_loss = 0.0
    y_true, y_pred = [], []

    for input_ids, token_type_ids, attention_mask, y in val_loader:
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        y = y.to(device).long().view(-1)

        outputs, pooled, logits = model([input_ids, token_type_ids, attention_mask])

        total_loss += F.cross_entropy(logits, y).item()
        pred = torch.argmax(logits, dim=-1)

        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(val_loader))

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix -> same output file name
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["benign", "malware"],
        yticklabels=["benign", "malware"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%"
        .format(avg_loss, acc * 100, prec * 100, rec * 100, f1 * 100)
    )

    return acc, prec, rec, f1


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load CSV
    df = load_csv_as_binary(CSV_PATH)

    # 2) Train/Val split (stratified)
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    # 3) Tokenize
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    MAX_LEN = 128  # adjust if your model expects a different fixed length

    x1_train, x2_train, x3_train = encode_urls(tokenizer, train_df["url"], max_len=MAX_LEN)
    y_train = torch.tensor(train_df["label"].values, dtype=torch.long)

    x1_val, x2_val, x3_val = encode_urls(tokenizer, val_df["url"], max_len=MAX_LEN)
    y_val = torch.tensor(val_df["label"].values, dtype=torch.long)

    # 4) DataLoaders (DO NOT .to(DEVICE) inside TensorDataset; move in loop)
    BATCH_SIZE = 4
    train_data = TensorDataset(x1_train, x2_train, x3_train, y_train)
    val_data = TensorDataset(x1_val, x2_val, x3_val, y_val)

    train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=BATCH_SIZE)

    # 5) Model + Optimizer
    model = Model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    best_acc = 0.0
    NUM_EPOCHS = 3

    for epoch in range(1, NUM_EPOCHS + 1):
        train_one_epoch(model, DEVICE, train_loader, optimizer, epoch)
        acc, prec, rec, f1 = validation(model, DEVICE, val_loader, cm_path=CM_PATH)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_PATH)

        print(f"acc is: {acc:.4f}, best acc is {best_acc:.4f}\n")


if __name__ == "__main__":
    main()

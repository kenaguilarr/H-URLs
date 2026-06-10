import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from url_dataprocessing import dataPreprocess_charbert, spiltDatast_charbert
from Model_PMA import CharBertModel


def train(model, device, train_loader, optimizer, scaler, epoch, accumulation_steps=1):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (x1, x2, x3, x4, x5, x6, y) in enumerate(train_loader):
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        x4 = x4.to(device)
        x5 = x5.to(device)
        x6 = x6.to(device)
        y = y.to(device).long().view(-1)

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            _, _, logits = model([x1, x2, x3, x4, x5, x6])
            loss = F.cross_entropy(logits, y)
            scaled_loss = loss / accumulation_steps

        scaler.scale(scaled_loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(train_loader):
            print(
                f"Train Epoch: {epoch} [{(batch_idx + 1) * len(x1)}/{len(train_loader.dataset)} "
                f"({100. * (batch_idx + 1) / len(train_loader):.2f}%)] Loss: {loss.item():.6f}"
            )


def validation(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for x1, x2, x3, x4, x5, x6, y in test_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            x4 = x4.to(device)
            x5 = x5.to(device)
            x6 = x6.to(device)
            y = y.to(device).long().view(-1)

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                _, _, logits = model([x1, x2, x3, x4, x5, x6])
                test_loss += F.cross_entropy(logits, y).item()

            pred = logits.argmax(dim=-1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    test_loss /= len(test_loader)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["benign", "malware"],
        yticklabels=["benign", "malware"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%".format(
            test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100
        )
    )

    return accuracy, precision, recall, f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids = []
    input_types = []
    input_masks = []
    char_ids = []
    start_ids = []
    end_ids = []
    label = []

    dataPreprocess_charbert(
        "Data/Raw_Dataset_QR/url/url_train/Train_Benign.xlsx",
        input_ids,
        input_types,
        input_masks,
        char_ids,
        start_ids,
        end_ids,
        label,
        0,
    )
    dataPreprocess_charbert(
        "Data/Raw_Dataset_QR/url/url_train/Train_Malicious.xlsx",
        input_ids,
        input_types,
        input_masks,
        char_ids,
        start_ids,
        end_ids,
        label,
        1,
    )

    (
        input_ids_train,
        input_types_train,
        input_masks_train,
        char_ids_train,
        start_ids_train,
        end_ids_train,
        y_train,
        input_ids_val,
        input_types_val,
        input_masks_val,
        char_ids_val,
        start_ids_val,
        end_ids_val,
        y_val,
    ) = spiltDatast_charbert(
        input_ids,
        input_types,
        input_masks,
        char_ids,
        start_ids,
        end_ids,
        label,
        train_ratio=0.95,
    )

    batch_size = 4
    accumulation_steps = 1
    train_data = TensorDataset(
        torch.tensor(input_ids_train, dtype=torch.long),
        torch.tensor(input_types_train, dtype=torch.long),
        torch.tensor(input_masks_train, dtype=torch.long),
        torch.tensor(char_ids_train, dtype=torch.long),
        torch.tensor(start_ids_train, dtype=torch.long),
        torch.tensor(end_ids_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.long).view(-1),
    )
    train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)

    val_data = TensorDataset(
        torch.tensor(input_ids_val, dtype=torch.long),
        torch.tensor(input_types_val, dtype=torch.long),
        torch.tensor(input_masks_val, dtype=torch.long),
        torch.tensor(char_ids_val, dtype=torch.long),
        torch.tensor(start_ids_val, dtype=torch.long),
        torch.tensor(end_ids_val, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.long).view(-1),
    )
    val_loader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)

    model = CharBertModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_acc = 0.0
    num_epochs = 3
    path = "urlmodel_charbert.pth"

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, scaler, epoch, accumulation_steps=accumulation_steps)
        acc, precision, recall, f1 = validation(model, device, val_loader)

        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), path)
        print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))


if __name__ == "__main__":
    main()

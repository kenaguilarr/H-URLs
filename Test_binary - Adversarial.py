import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from url_dataprocessing import dataPreprocessFromCSV
from Model_PMA import Model


def test_binary(model, device, test_loader,
                cm_path="confusion_matrix.png",
                roc_path="roc_curve.png",
                results_path="results.txt"):
    model.eval()

    test_loss = 0.0
    y_true, y_pred, y_probs = [], [], []

    with torch.inference_mode():
        for x1, x2, x3, y in test_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            y = y.to(device).view(-1).long()   # (B,)

            outputs, pooled, logits = model([x1, x2, x3])  # logits: (B, 2)

            loss = F.cross_entropy(logits, y)
            test_loss += loss.item()

            probs = torch.softmax(logits, dim=1)[:, 1]     # P(class=1)
            preds = torch.argmax(logits, dim=1)            # (B,)

            y_true.extend(y.detach().cpu().numpy().ravel().tolist())
            y_pred.extend(preds.detach().cpu().numpy().ravel().tolist())
            y_probs.extend(probs.detach().cpu().numpy().ravel().tolist())

    test_loss /= max(1, len(test_loader))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["benign", "malware"],
                yticklabels=["benign", "malware"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    plt.close()

    # Save results
    results_array = np.column_stack([y_true, y_pred, y_probs])
    header_text = "true_label\tpred_label\tprob_positive_class"
    np.savetxt(results_path, results_array, fmt="%.6f", delimiter="\t", header=header_text)

    print(f"Test set: Avg loss: {test_loss:.4f}, "
          f"Acc: {accuracy*100:.2f}%, Prec: {precision*100:.2f}%, "
          f"Recall: {recall*100:.2f}%, F1: {f1*100:.2f}%, "
          f"AUC: {roc_auc:.4f}")

    return accuracy, precision, recall, f1, roc_auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids, input_types, input_masks, label = [], [], [], []

    # ✅ use your uploaded test data path here
    test_csv = "Data/Grambedding_dataset_Adversarial_Full/test_aug_mode2.csv"
    dataPreprocessFromCSV(test_csv, input_ids, input_types, input_masks, label)

    # ✅ keep dataset on CPU; move batch to GPU inside test loop
    test_data = TensorDataset(
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(input_types, dtype=torch.long),
        torch.tensor(input_masks, dtype=torch.long),
        torch.tensor(label, dtype=torch.long)
    )

    test_loader = DataLoader(
        test_data,
        sampler=SequentialSampler(test_data),
        batch_size=4
    )

    model = Model().to(device)
    model.load_state_dict(torch.load("model.pth", map_location=device))

    test_binary(model, device, test_loader)


if __name__ == "__main__":
    main()

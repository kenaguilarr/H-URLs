import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from pathlib import Path

from html_dataprocessing import dataPreprocessFromHTMLFolder
from Model_PMA import Model


HTML_EXTENSIONS = (".html", ".htm", ".txt")


def count_html_inputs(folder_path):
    folder = Path(folder_path)
    if not folder.exists():
        return 0
    return sum(1 for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in HTML_EXTENSIONS)


def test_binary(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for x1, x2, x3, y in test_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            y = y.to(device).long().view(-1)

            outputs, pooled, logits = model([x1, x2, x3])

            test_loss += F.cross_entropy(logits, y).item()

            pred = logits.argmax(dim=-1)

            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            y_probs.extend(torch.softmax(logits, dim=1).cpu().numpy()[:, 1])

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
    plt.close()

                                                      
    if len(set(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.savefig("html_roc_curve.png")
        plt.close()
    else:
        print("Skipping ROC curve: test set does not contain both classes.")

    results_array = np.column_stack((y_true, y_pred, y_probs))
    header_text = "True label, Predicted label, Predicted Probability"
    np.savetxt("results.txt", results_array, fmt="%1.6f", delimiter="\t", header=header_text)

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
    labels = []

    benign_dir = "Data/Raw_Dataset_QR/CrossData/html/Mendeley_benign"
    malicious_dir = "Data/Raw_Dataset_QR/CrossData/html/Mendeley_malicious"

    benign_count = count_html_inputs(benign_dir)
    malicious_count = count_html_inputs(malicious_dir)
    print(f"Found {benign_count} benign HTML input files in {benign_dir}")
    print(f"Found {malicious_count} malicious HTML input files in {malicious_dir}")
    if benign_count == 0 and malicious_count == 0:
        raise FileNotFoundError("No .html, .htm, or .txt files found for HTML testing.")

    dataPreprocessFromHTMLFolder(benign_dir, input_ids, input_types, input_masks, labels, urltype=0, max_chars=50000)
    dataPreprocessFromHTMLFolder(malicious_dir, input_ids, input_types, input_masks, labels, urltype=1, max_chars=50000)

    batch_size = 4
    test_data = TensorDataset(
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(input_types, dtype=torch.long),
        torch.tensor(input_masks, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long).view(-1),
    )
    test_loader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=batch_size)

    model = Model().to(device)
    model.load_state_dict(torch.load("htmlmodel.pth", map_location=device))

    test_binary(model, device, test_loader)


if __name__ == "__main__":
    main()

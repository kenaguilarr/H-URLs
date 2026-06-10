import argparse
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader, SequentialSampler
import matplotlib.pyplot as plt
import seaborn as sns

from fusion_dataprocessing import load_raw_fusion_pairs, encode_fusion_samples, to_tensor_dataset
from fusion_model import FusionModel, load_fusion_checkpoint


def test_binary(model, device, test_loader):
    model.eval()
    test_loss = 0.0
    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for u_ids, u_types, u_masks, u_char_ids, u_start_ids, u_end_ids, h_ids, h_types, h_masks, y in test_loader:
            u_ids = u_ids.to(device)
            u_types = u_types.to(device)
            u_masks = u_masks.to(device)
            u_char_ids = u_char_ids.to(device)
            u_start_ids = u_start_ids.to(device)
            u_end_ids = u_end_ids.to(device)
            h_ids = h_ids.to(device)
            h_types = h_types.to(device)
            h_masks = h_masks.to(device)
            y = y.to(device).long().view(-1)

            outputs = model(
                url_inputs=[u_ids, u_types, u_masks, u_char_ids, u_start_ids, u_end_ids],
                html_inputs=[h_ids, h_types, h_masks],
            )
            logits = outputs["fusion_logits"]

            test_loss += F.cross_entropy(logits, y).item()
            pred = logits.argmax(dim=-1)

            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())
            y_prob.extend(torch.softmax(logits, dim=1).cpu().numpy()[:, 1])

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
    plt.title("Fusion Confusion Matrix")
    plt.savefig("fusion_confusion_matrix.png")
    plt.close()

    if len(set(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Fusion ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig("fusion_roc_curve.png")
        plt.close()
    else:
        print("Skipping ROC curve: test set does not contain both classes.")

    arr = np.column_stack((y_true, y_pred, y_prob))
    np.savetxt(
        "fusion_results.txt",
        arr,
        fmt="%1.6f",
        delimiter="\t",
        header="True label, Predicted label, Predicted Probability",
    )

    print(
        "Fusion Test: loss={:.4f}, acc={:.2f}%, precision={:.2f}%, recall={:.2f}%, f1={:.2f}%".format(
            test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100
        )
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Test the CharBERT+HTML fusion model.")
    parser.add_argument("--checkpoint-path", type=str, default="fusion_model.pth")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--pad-size", type=int, default=200)
    parser.add_argument("--max-html-chars", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_raw_fusion_pairs(
        raw_root="Data/Raw_Dataset_QR",
        seed=args.seed,
        max_html_chars=args.max_html_chars,
    )
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    encoded = encode_fusion_samples(samples, pad_size=args.pad_size)
    test_ds = to_tensor_dataset(encoded)

    test_loader = DataLoader(
        test_ds,
        sampler=SequentialSampler(test_ds),
        batch_size=args.batch_size,
    )

    model = FusionModel(
        url_ckpt="urlmodel_charbert.pth",
        html_ckpt="htmlmodel.pth",
        freeze_encoders=True,
    ).to(device)
    load_fusion_checkpoint(model, args.checkpoint_path)

    test_binary(model, device, test_loader)


if __name__ == "__main__":
    main()

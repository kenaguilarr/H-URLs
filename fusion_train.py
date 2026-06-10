import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset

from fusion_dataprocessing import build_fusion_train_val_datasets
from fusion_model import FusionModel


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def train_one_epoch(model, device, loader, optimizer, epoch,
                    w_url=1.0, w_html=1.0, w_fusion=1.0):
    model.train()

    for batch_idx, batch in enumerate(loader):
        (u_ids, u_types, u_masks, u_char_ids, u_start_ids, u_end_ids, h_ids, h_types, h_masks, y) = batch

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

        optimizer.zero_grad()

        outputs = model(
            url_inputs=[u_ids, u_types, u_masks, u_char_ids, u_start_ids, u_end_ids],
            html_inputs=[h_ids, h_types, h_masks],
        )

        loss_url = F.cross_entropy(outputs["url_logits"], y)
        loss_html = F.cross_entropy(outputs["html_logits"], y)
        loss_fusion = F.cross_entropy(outputs["fusion_logits"], y)
        loss = w_url * loss_url + w_html * loss_html + w_fusion * loss_fusion
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 50 == 0:
            print(
                f"Epoch {epoch} Step {batch_idx + 1}/{len(loader)} "
                f"Loss={loss.item():.4f} "
                f"(url={loss_url.item():.4f}, html={loss_html.item():.4f}, fusion={loss_fusion.item():.4f})"
            )


def evaluate(model, device, loader):
    model.eval()

    y_true = []
    y_pred_url = []
    y_pred_html = []
    y_pred_fusion = []

    with torch.no_grad():
        for batch in loader:
            (u_ids, u_types, u_masks, u_char_ids, u_start_ids, u_end_ids, h_ids, h_types, h_masks, y) = batch

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

            pred_url = outputs["url_logits"].argmax(dim=-1)
            pred_html = outputs["html_logits"].argmax(dim=-1)
            pred_fusion = outputs["fusion_logits"].argmax(dim=-1)

            y_true.extend(y.cpu().tolist())
            y_pred_url.extend(pred_url.cpu().tolist())
            y_pred_html.extend(pred_html.cpu().tolist())
            y_pred_fusion.extend(pred_fusion.cpu().tolist())

    return {
        "url": compute_metrics(y_true, y_pred_url),
        "html": compute_metrics(y_true, y_pred_html),
        "fusion": compute_metrics(y_true, y_pred_fusion),
    }


def _format_metrics(name, m):
    return (
        f"{name}: acc={m['accuracy'] * 100:.2f}% "
        f"prec={m['precision'] * 100:.2f}% "
        f"rec={m['recall'] * 100:.2f}% "
        f"f1={m['f1'] * 100:.2f}%"
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Train the CharBERT+HTML fusion model.")
    parser.add_argument("--train-ratio", type=float, default=0.95)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--pad-size", type=int, default=200)
    parser.add_argument("--max-html-chars", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--checkpoint-path", type=str, default="fusion_model.pth")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def _maybe_limit_dataset(dataset, max_samples):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def main():
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, val_ds = build_fusion_train_val_datasets(
        raw_root="Data/Raw_Dataset_QR",
        pad_size=args.pad_size,
        train_ratio=args.train_ratio,
        seed=args.seed,
        max_html_chars=args.max_html_chars,
    )
    train_ds = _maybe_limit_dataset(train_ds, args.max_train_samples)
    val_ds = _maybe_limit_dataset(val_ds, args.max_val_samples)

    train_loader = DataLoader(train_ds, sampler=RandomSampler(train_ds), batch_size=args.batch_size)
    val_loader = DataLoader(val_ds, sampler=SequentialSampler(val_ds), batch_size=args.batch_size)

    model = FusionModel(
        url_ckpt="urlmodel_charbert.pth",
        html_ckpt="htmlmodel.pth",
        freeze_encoders=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    best_fusion_acc = -1.0

    for epoch in range(1, args.num_epochs + 1):
        train_one_epoch(model, device, train_loader, optimizer, epoch)

        metrics = evaluate(model, device, val_loader)
        print(_format_metrics("URL", metrics["url"]))
        print(_format_metrics("HTML", metrics["html"]))
        print(_format_metrics("FUSION", metrics["fusion"]))

        fusion_acc = metrics["fusion"]["accuracy"]
        if fusion_acc > best_fusion_acc:
            best_fusion_acc = fusion_acc
            torch.save(model.state_dict(), args.checkpoint_path)

        print(f"Best fusion acc so far: {best_fusion_acc * 100:.2f}%\n")


if __name__ == "__main__":
    main()

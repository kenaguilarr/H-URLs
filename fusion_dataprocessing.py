import os

import numpy as np
import pandas as pd
import torch
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import TensorDataset
from url_dataprocessing import CharbertInput


BERT_PATH = "charbert-bert-wiki/charbert-bert-wiki/"
VOCAB_PATH = os.path.join(BERT_PATH, "vocab.txt")


def _build_tokenizer():
    return BertTokenizer(vocab_file=VOCAB_PATH)


def _encode_text(tokenizer, text, pad_size=200):
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    ids = tokenizer.convert_tokens_to_ids(tokens)
    types = [0] * len(ids)
    masks = [1] * len(ids)

    if len(ids) < pad_size:
        pad_len = pad_size - len(ids)
        ids = ids + [0] * pad_len
        types = types + [0] * pad_len
        masks = masks + [0] * pad_len
    else:
        ids = ids[:pad_size]
        types = types[:pad_size]
        masks = masks[:pad_size]

    return ids, types, masks


def _encode_url_text_charbert(tokenizer, text, pad_size=200):
    tokens = tokenizer.tokenize(text)
    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    ids = tokenizer.convert_tokens_to_ids(tokens)
    types = [0] * len(ids)
    masks = [1] * len(ids)

    if len(ids) < pad_size:
        pad_len = pad_size - len(ids)
        ids = ids + [0] * pad_len
        types = types + [1] * pad_len
        masks = masks + [0] * pad_len
    else:
        ids = ids[:pad_size]
        types = types[:pad_size]
        masks = masks[:pad_size]

    char_ids, start_ids, end_ids = CharbertInput(ids, tokenizer=tokenizer)
    return ids, types, masks, char_ids, start_ids, end_ids


def _normalize_html_key(value):
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        if np.isnan(value):
            return None
        return str(int(value))
    text = str(value).strip()
    if not text:
        return None
    base = os.path.basename(text)
    stem, _ = os.path.splitext(base)
    return stem if stem else base


def _normalize_class_subdirs(class_subdir):
    if class_subdir is None:
        return []
    if isinstance(class_subdir, (list, tuple)):
        return [str(item).strip() for item in class_subdir if str(item).strip()]
    text = str(class_subdir).strip()
    return [text] if text else []


def _resolve_html_path(raw_root, html_path, class_subdir):
    if html_path is None:
        return None

    p = str(html_path).strip()
    if not p:
        return None

    key = _normalize_html_key(html_path)

    class_subdirs = _normalize_class_subdirs(class_subdir)

    candidates = [
        p,
        os.path.join(raw_root, p),
        os.path.join(raw_root, "html", p),
    ]
    for subdir in class_subdirs:
        candidates.append(os.path.join(raw_root, "html", subdir, p))
    if key:
        for subdir in class_subdirs:
            candidates.extend([
                os.path.join(raw_root, "html", subdir, key + ".txt"),
                os.path.join(raw_root, "html", subdir, key + ".html"),
                os.path.join(raw_root, "html", subdir, key + ".htm"),
            ])
        candidates.extend([
            os.path.join(raw_root, "html", key + ".txt"),
            os.path.join(raw_root, "html", key + ".html"),
            os.path.join(raw_root, "html", key + ".htm"),
        ])

    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _load_pairs_from_xlsx(xlsx_path, label, raw_root, class_subdir, max_html_chars=50000, encoding="utf-8"):
    try:
        df = pd.read_excel(xlsx_path, header=None)
    except ImportError as e:
        raise ImportError(
            "Reading Excel requires openpyxl. Install it with: pip install openpyxl"
        ) from e

    if df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {xlsx_path} (html_path, url).")

    pairs = []
    for _, row in df.iterrows():
        html_cell = row.iloc[0]
        url_cell = None
        for cell in row.tolist():
            if pd.isna(cell):
                continue
            cell_text = str(cell).strip()
            if cell_text.lower().startswith(("http://", "https://")):
                url_cell = cell
                break

        if pd.isna(html_cell) or url_cell is None:
            continue

        url_text = str(url_cell).strip()
        if not url_text:
            continue

                                                 
        if isinstance(url_text, str) and not url_text.lower().startswith(("http://", "https://")):
            continue

        html_fp = _resolve_html_path(raw_root, html_cell, class_subdir=class_subdir)
        if html_fp is None:
            continue

        try:
            with open(html_fp, "r", encoding=encoding, errors="ignore") as f:
                html_text = f.read()
        except Exception:
            continue

        if max_html_chars is not None:
            html_text = html_text[:max_html_chars]
        if not html_text.strip():
            continue

        pairs.append((url_text, html_text, label))

    return pairs


def load_raw_fusion_pairs(
    raw_root="Data/Raw_Dataset_QR",
    benign_xlsx="Data/Raw_Dataset_QR/url/url_train/Train_Benign.xlsx",
    malicious_xlsx="Data/Raw_Dataset_QR/url/url_train/Train_Malicious.xlsx",
    seed=2020,
    max_html_chars=50000,
):
    benign_pairs = _load_pairs_from_xlsx(
        benign_xlsx,
        label=0,
        raw_root=raw_root,
        class_subdir=("html_benign_train", "benign", "c_benign", "Train_Benign"),
        max_html_chars=max_html_chars,
    )
    malicious_pairs = _load_pairs_from_xlsx(
        malicious_xlsx,
        label=1,
        raw_root=raw_root,
        class_subdir=("html_malicious_train", "malicious", "c_malicious", "Train_Malicious"),
        max_html_chars=max_html_chars,
    )

    samples = benign_pairs + malicious_pairs
    rng = np.random.RandomState(seed)
    rng.shuffle(samples)

    print(f"Fusion benign pairs: {len(benign_pairs)}")
    print(f"Fusion malicious pairs: {len(malicious_pairs)}")
    print(f"Fusion total pairs: {len(samples)}")

    return samples


def encode_fusion_samples(samples, pad_size=200):
    tokenizer = _build_tokenizer()

    url_ids, url_types, url_masks = [], [], []
    url_char_ids, url_start_ids, url_end_ids = [], [], []
    html_ids, html_types, html_masks = [], [], []
    labels = []

    for url_text, html_text, y in samples:
        u_ids, u_types, u_masks, u_char_ids, u_start_ids, u_end_ids = _encode_url_text_charbert(
            tokenizer, url_text, pad_size=pad_size
        )
        h_ids, h_types, h_masks = _encode_text(tokenizer, html_text, pad_size=pad_size)

        url_ids.append(u_ids)
        url_types.append(u_types)
        url_masks.append(u_masks)
        url_char_ids.append(u_char_ids)
        url_start_ids.append(u_start_ids)
        url_end_ids.append(u_end_ids)

        html_ids.append(h_ids)
        html_types.append(h_types)
        html_masks.append(h_masks)

        labels.append([y])

    return (
        np.array(url_ids),
        np.array(url_types),
        np.array(url_masks),
        np.array(url_char_ids),
        np.array(url_start_ids),
        np.array(url_end_ids),
        np.array(html_ids),
        np.array(html_types),
        np.array(html_masks),
        np.array(labels),
    )


def spiltDatast_fusion(url_ids, url_types, url_masks, url_char_ids, url_start_ids, url_end_ids,
                       html_ids, html_types, html_masks, labels,
                       train_ratio=0.8, split_ratio=None, seed=2020):
    random_order = list(range(len(labels)))
    np.random.seed(seed)
    np.random.shuffle(random_order)
    print(random_order[:10])

    effective_ratio = train_ratio if split_ratio is None else split_ratio
    split_idx = int(len(labels) * effective_ratio)
    if len(labels) > 1:
        split_idx = max(1, min(split_idx, len(labels) - 1))
    train_idx = random_order[:split_idx]
    test_idx = random_order[split_idx:]

    url_ids_train = np.array([url_ids[i] for i in train_idx])
    url_types_train = np.array([url_types[i] for i in train_idx])
    url_masks_train = np.array([url_masks[i] for i in train_idx])
    url_char_ids_train = np.array([url_char_ids[i] for i in train_idx])
    url_start_ids_train = np.array([url_start_ids[i] for i in train_idx])
    url_end_ids_train = np.array([url_end_ids[i] for i in train_idx])
    html_ids_train = np.array([html_ids[i] for i in train_idx])
    html_types_train = np.array([html_types[i] for i in train_idx])
    html_masks_train = np.array([html_masks[i] for i in train_idx])
    y_train = np.array([labels[i] for i in train_idx])

    url_ids_test = np.array([url_ids[i] for i in test_idx])
    url_types_test = np.array([url_types[i] for i in test_idx])
    url_masks_test = np.array([url_masks[i] for i in test_idx])
    url_char_ids_test = np.array([url_char_ids[i] for i in test_idx])
    url_start_ids_test = np.array([url_start_ids[i] for i in test_idx])
    url_end_ids_test = np.array([url_end_ids[i] for i in test_idx])
    html_ids_test = np.array([html_ids[i] for i in test_idx])
    html_types_test = np.array([html_types[i] for i in test_idx])
    html_masks_test = np.array([html_masks[i] for i in test_idx])
    y_test = np.array([labels[i] for i in test_idx])

    print("url_ids_train.shape:" + str(url_ids_train.shape))
    print("url_types_train.shape:" + str(url_types_train.shape))
    print("url_masks_train.shape:" + str(url_masks_train.shape))
    print("url_char_ids_train.shape:" + str(url_char_ids_train.shape))
    print("url_start_ids_train.shape:" + str(url_start_ids_train.shape))
    print("url_end_ids_train.shape:" + str(url_end_ids_train.shape))
    print("html_ids_train.shape:" + str(html_ids_train.shape))
    print("html_types_train.shape:" + str(html_types_train.shape))
    print("html_masks_train.shape:" + str(html_masks_train.shape))
    print("y_train.shape:" + str(y_train.shape))

    print("url_ids_test.shape:" + str(url_ids_test.shape))
    print("url_types_test.shape:" + str(url_types_test.shape))
    print("url_masks_test.shape:" + str(url_masks_test.shape))
    print("url_char_ids_test.shape:" + str(url_char_ids_test.shape))
    print("url_start_ids_test.shape:" + str(url_start_ids_test.shape))
    print("url_end_ids_test.shape:" + str(url_end_ids_test.shape))
    print("html_ids_test.shape:" + str(html_ids_test.shape))
    print("html_types_test.shape:" + str(html_types_test.shape))
    print("html_masks_test.shape:" + str(html_masks_test.shape))
    print("y_test.shape:" + str(y_test.shape))

    return (
        url_ids_train, url_types_train, url_masks_train, url_char_ids_train, url_start_ids_train, url_end_ids_train,
        html_ids_train, html_types_train, html_masks_train, y_train,
        url_ids_test, url_types_test, url_masks_test, url_char_ids_test, url_start_ids_test, url_end_ids_test,
        html_ids_test, html_types_test, html_masks_test, y_test
    )


def to_tensor_dataset(split_tuple):
    u_ids, u_types, u_masks, u_char_ids, u_start_ids, u_end_ids, h_ids, h_types, h_masks, y = split_tuple

    return TensorDataset(
        torch.tensor(u_ids, dtype=torch.long),
        torch.tensor(u_types, dtype=torch.long),
        torch.tensor(u_masks, dtype=torch.long),
        torch.tensor(u_char_ids, dtype=torch.long),
        torch.tensor(u_start_ids, dtype=torch.long),
        torch.tensor(u_end_ids, dtype=torch.long),
        torch.tensor(h_ids, dtype=torch.long),
        torch.tensor(h_types, dtype=torch.long),
        torch.tensor(h_masks, dtype=torch.long),
        torch.tensor(y, dtype=torch.long).view(-1),
    )


def build_fusion_train_val_datasets(raw_root="Data/Raw_Dataset_QR", pad_size=200, train_ratio=0.8,
                                    split_ratio=None, seed=2020, max_html_chars=50000):
    samples = load_raw_fusion_pairs(
        raw_root=raw_root,
        benign_xlsx=os.path.join(raw_root, "url", "url_train", "Train_Benign.xlsx"),
        malicious_xlsx=os.path.join(raw_root, "url", "url_train", "Train_Malicious.xlsx"),
        seed=seed,
        max_html_chars=max_html_chars,
    )

    encoded = encode_fusion_samples(samples, pad_size=pad_size)
    (url_ids_train, url_types_train, url_masks_train, url_char_ids_train, url_start_ids_train, url_end_ids_train,
     html_ids_train, html_types_train, html_masks_train, y_train,
     url_ids_test, url_types_test, url_masks_test, url_char_ids_test, url_start_ids_test, url_end_ids_test,
     html_ids_test, html_types_test, html_masks_test, y_test) = spiltDatast_fusion(
        *encoded, train_ratio=train_ratio, split_ratio=split_ratio, seed=seed
    )

    train_ds = to_tensor_dataset((
        url_ids_train, url_types_train, url_masks_train, url_char_ids_train, url_start_ids_train, url_end_ids_train,
        html_ids_train, html_types_train, html_masks_train, y_train
    ))
    val_ds = to_tensor_dataset((
        url_ids_test, url_types_test, url_masks_test, url_char_ids_test, url_start_ids_test, url_end_ids_test,
        html_ids_test, html_types_test, html_masks_test, y_test
    ))

    return train_ds, val_ds


def build_fusion_train_test_datasets(raw_root="Data/Raw_Dataset_QR", pad_size=200, train_ratio=0.8,
                                     split_ratio=None, seed=2020, max_html_chars=50000):
    return build_fusion_train_val_datasets(
        raw_root=raw_root,
        pad_size=pad_size,
        train_ratio=train_ratio,
        split_ratio=split_ratio,
        seed=seed,
        max_html_chars=max_html_chars,
    )

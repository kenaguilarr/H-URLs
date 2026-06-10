import os
import glob
import collections
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np


                              
                            
                              
def dataPreprocess_bert(filename, input_ids, input_types, input_masks, label, urltype):
    pad_size = 200
    bert_path = "charbert-bert-wiki/charbert-bert-wiki/"
    tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")

    with open(filename, encoding='latin-1', errors='ignore') as f:
        for i, l in tqdm(enumerate(f), desc=f"Reading lines from {filename}"):
            x1 = l.strip()
            x1 = tokenizer.tokenize(x1)
            tokens = ["[CLS]"] + x1 + ["[SEP]"]

            ids = tokenizer.convert_tokens_to_ids(tokens)
            types = [0] * len(ids)
            masks = [1] * len(ids)

            if len(ids) < pad_size:
                pad_len = pad_size - len(ids)
                ids = ids + [0] * pad_len
                masks = masks + [0] * pad_len
                types = types + [0] * pad_len                                          
            else:
                ids = ids[:pad_size]
                masks = masks[:pad_size]
                types = types[:pad_size]

            input_ids.append(ids)
            input_types.append(types)
            input_masks.append(masks)

            assert len(ids) == len(masks) == len(types) == pad_size

            label.append([1] if urltype == 1 else [0])


                              
                                  
                              
def dataPreprocessFromHTMLFolder(folder_path, input_ids, input_types, input_masks, label, urltype,
                                pad_size=200, encoding='utf-8', max_chars=None):
    bert_path = "charbert-bert-wiki/charbert-bert-wiki/"
    tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")

    html_files = sorted(
        glob.glob(os.path.join(folder_path, "**", "*.html"), recursive=True) +
        glob.glob(os.path.join(folder_path, "**", "*.htm"), recursive=True) +
        glob.glob(os.path.join(folder_path, "**", "*.txt"), recursive=True)
    )

    for fp in tqdm(html_files, desc=f"Reading HTML files from {folder_path}"):
        try:
            with open(fp, "r", encoding=encoding, errors="ignore") as f:
                text = f.read()
        except Exception:
            continue

        if max_chars is not None:
            text = text[:max_chars]

        tokens = tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        ids = tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * len(ids)
        masks = [1] * len(ids)

        if len(ids) < pad_size:
            pad_len = pad_size - len(ids)
            ids = ids + [0] * pad_len
            masks = masks + [0] * pad_len
            types = types + [0] * pad_len
        else:
            ids = ids[:pad_size]
            masks = masks[:pad_size]
            types = types[:pad_size]

        input_ids.append(ids)
        input_types.append(types)
        input_masks.append(masks)

        label.append([1] if urltype == 1 else [0])

        assert len(ids) == len(masks) == len(types) == pad_size


                              
                          
                              
def dataPreprocessFromCSV(filename, input_ids, input_types, input_masks, label=None, is_CharBert=False):
    pad_size = 200
    bert_path = "charbert-bert-wiki/charbert-bert-wiki/"
    tokenizer = BertTokenizer(vocab_file=bert_path + "vocab.txt")

    data = pd.read_csv(filename, encoding='utf-8')

    for i, row in tqdm(data.iterrows(), total=len(data), desc=f"Reading CSV rows from {filename}"):
        x1 = row['url']
        x1 = tokenizer.tokenize(x1)
        tokens = ["[CLS]"] + x1 + ["[SEP]"]

        ids = tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * len(ids)
        masks = [1] * len(ids)

        if len(ids) < pad_size:
            pad_len = pad_size - len(ids)
            ids = ids + [0] * pad_len
            masks = masks + [0] * pad_len
            types = types + [0] * pad_len
        else:
            ids = ids[:pad_size]
            masks = masks[:pad_size]
            types = types[:pad_size]

        input_ids.append(ids)
        input_types.append(types)
        input_masks.append(masks)

        assert len(ids) == len(masks) == len(types) == pad_size

        y = row['label']
        if y == 'malicious':
            label.append([1])
        elif y == 'benign':
            label.append([0])


                              
       
                              
def spiltDatast_bert(input_ids, input_types, input_masks, label, train_ratio=0.8):                                
    random_order = list(range(len(input_ids)))
    np.random.seed(2020)
    np.random.shuffle(random_order)

    split_idx = int(len(input_ids) * train_ratio)

    input_ids_train = np.array([input_ids[i] for i in random_order[:split_idx]])
    input_types_train = np.array([input_types[i] for i in random_order[:split_idx]])
    input_masks_train = np.array([input_masks[i] for i in random_order[:split_idx]])
    y_train = np.array([label[i] for i in random_order[:split_idx]])

    input_ids_test = np.array([input_ids[i] for i in random_order[split_idx:]])
    input_types_test = np.array([input_types[i] for i in random_order[split_idx:]])
    input_masks_test = np.array([input_masks[i] for i in random_order[split_idx:]])
    y_test = np.array([label[i] for i in random_order[split_idx:]])

    print("input_ids_train.shape:", input_ids_train.shape)
    print("input_types_train.shape:", input_types_train.shape)
    print("input_masks_train.shape:", input_masks_train.shape)
    print("y_train.shape:", y_train.shape)

    print("input_ids_test.shape:", input_ids_test.shape)
    print("input_types_test.shape:", input_types_test.shape)
    print("input_masks_test.shape:", input_masks_test.shape)
    print("y_test.shape:", y_test.shape)

    return (input_ids_train, input_types_train, input_masks_train, y_train,
            input_ids_test, input_types_test, input_masks_test, y_test)
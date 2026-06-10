import collections
import os
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
import pandas as pd
import numpy as np


CHARBERT_MODEL_DIR = os.path.join("charbert-bert-wiki", "charbert-bert-wiki")
CHARBERT_VOCAB_FILE = os.path.join(CHARBERT_MODEL_DIR, "vocab.txt")
CHARBERT_CHAR_VOCAB_FILE = os.path.join("character_bert_wiki", "vocab.txt")
CHARBERT_CHAR_VOCAB_SIZE = 1001
_CHAR_TO_IDS = None


def _build_tokenizer():
    return BertTokenizer(vocab_file=CHARBERT_VOCAB_FILE)


def _iter_urls_from_source(filename):
    ext = os.path.splitext(filename)[1].lower()
    if ext in [".xlsx", ".xls"]:
        try:
            df = pd.read_excel(filename, header=None)
        except ImportError as e:
            raise ImportError(
                "Reading Excel requires openpyxl. Install it with: pip install openpyxl"
            ) from e
        if df.shape[1] < 2:
            raise ValueError(f"Expected at least 2 columns in {filename} (html_path, url).")
        for url in df.iloc[:, 1].tolist():
            if pd.notna(url):
                text = str(url).strip()
                if text and text.lower().startswith(("http://", "https://")):
                    yield text
        return

    with open(filename, encoding='latin-1', errors='ignore') as f:
        for line in f:
            text = line.strip()
            if text:
                yield text


def dataPreprocess_bert(filename, input_ids, input_types, input_masks, label, urltype):
    pad_size = 200
                                                           
                                                            
    tokenizer = _build_tokenizer()
    for i, x1 in tqdm(enumerate(_iter_urls_from_source(filename))):
            x1 = tokenizer.tokenize(x1)
            tokens = ["[CLS]"] + x1 + ["[SEP]"]

                                            
            ids = tokenizer.convert_tokens_to_ids(tokens)
            types = [0] * (len(ids))
            masks = [1] * len(ids)

                                            
            if len(ids) < pad_size:
                types = types + [1] * (pad_size - len(ids))                                        
                masks = masks + [0] * (pad_size - len(ids))
                ids = ids + [0] * (pad_size - len(ids))
            else:
                types = types[:pad_size]
                masks = masks[:pad_size]
                ids = ids[:pad_size]
            input_ids.append(ids)
            input_types.append(types)
            input_masks.append(masks)

                                                             
            assert len(ids) == len(masks) == len(types) == pad_size

            if urltype == 1:
                label.append([1])
            elif urltype == 0:
                label.append([0])
                                                  

def dataPreprocess_charbert(filename, input_ids, input_types, input_masks, char_ids, start_ids, end_ids, label,
                            urltype):
    pad_size = 200
                                                           
                                                            
    tokenizer = _build_tokenizer()
    for i, x1 in tqdm(enumerate(_iter_urls_from_source(filename))):
            x1 = tokenizer.tokenize(x1)
            tokens = ["[CLS]"] + x1 + ["[SEP]"]

                                        
            ids = tokenizer.convert_tokens_to_ids(tokens)
            types = [0] * (len(ids))
            masks = [1] * len(ids)

            if len(ids) < pad_size:
                types = types + [1] * (pad_size - len(ids))
                masks = masks + [0] * (pad_size - len(ids))
                ids = ids + [0] * (pad_size - len(ids))
            else:
                types = types[:pad_size]
                masks = masks[:pad_size]
                ids = ids[:pad_size]

            input_ids.append(ids)
            input_types.append(types)
            input_masks.append(masks)

            char, start, end = CharbertInput(ids, tokenizer=tokenizer)

            char_ids.append(char)
            start_ids.append(start)
            end_ids.append(end)
                                                             
            assert len(ids) == len(masks) == len(types) == pad_size
            if urltype == 1:
                label.append([1])
            elif urltype == 0:
                label.append([0])


def dataPreprocessFromCSV(filename, input_ids, input_types, input_masks, label = None, is_CharBert=False):
    pad_size = 200
    tokenizer = _build_tokenizer()

    data = pd.read_csv(filename, encoding='utf-8')
    char_ids = []
    start_ids = []
    end_ids = []
    for i, row in tqdm(data.iterrows(), total=len(data)):
        x1 = row['url']                                                                                
        x1 = tokenizer.tokenize(x1)
        tokens = ["[CLS]"] + x1 + ["[SEP]"]

                                        
        ids = tokenizer.convert_tokens_to_ids(tokens)
        types = [0] * (len(ids))
        masks = [1] * len(ids)

                                        
        if len(ids) < pad_size:
            types = types + [1] * (pad_size - len(ids))                                        
            masks = masks + [0] * (pad_size - len(ids))
            ids = ids + [0] * (pad_size - len(ids))
        else:
            types = types[:pad_size]
            masks = masks[:pad_size]
            ids = ids[:pad_size]
        input_ids.append(ids)
        input_types.append(types)
        input_masks.append(masks)

        if is_CharBert:
            char, start, end = CharbertInput(ids, tokenizer=tokenizer)
            char_ids.append(char)
            start_ids.append(start)
            end_ids.append(end)
        assert len(ids) == len(masks) == len(types) == pad_size

        y = row['label']
        if y == 'malicious':
            label.append([1])
        elif y == 'benign':
            label.append([0])
    if is_CharBert:
        return char_ids, start_ids, end_ids


def spiltDatast_bert(input_ids, input_types, input_masks, label, train_ratio=0.8):
                                  
    random_order = list(range(len(input_ids)))
    np.random.seed(2020)                
    np.random.shuffle(random_order)
    print(random_order[:10])

    split_idx = int(len(input_ids) * train_ratio)
    input_ids_train = np.array([input_ids[i] for i in random_order[:split_idx]])
    input_types_train = np.array([input_types[i] for i in random_order[:split_idx]])
    input_masks_train = np.array([input_masks[i] for i in random_order[:split_idx]])
    y_train = np.array([label[i] for i in random_order[:split_idx]])

    input_ids_test = np.array([input_ids[i] for i in random_order[split_idx:]])
    input_types_test = np.array([input_types[i] for i in random_order[split_idx:]])
    input_masks_test = np.array([input_masks[i] for i in random_order[split_idx:]])
    y_test = np.array([label[i] for i in random_order[split_idx:]])

    print("input_ids_train.shape:" + str(input_ids_train.shape))
    print("input_types_train.shape:" + str(input_types_train.shape))
    print("input_masks_train.shape:" + str(input_masks_train.shape))
    print("y_train.shape:" + str(y_train.shape))

    print("input_ids_test.shape:" + str(input_ids_test.shape))
    print("input_types_test.shape:" + str(input_types_test.shape))
    print("input_masks_test.shape:" + str(input_masks_test.shape))
    print("y_test.shape:" + str(y_test.shape))

    return input_ids_train, input_types_train, input_masks_train, y_train, input_ids_test, input_types_test, input_masks_test, y_test


def spiltDatast_charbert(input_ids, input_types, input_masks, char_ids, start_ids, end_ids, label, train_ratio=0.8):
                                  
    random_order = list(range(len(input_ids)))
    np.random.seed(2020)                
    np.random.shuffle(random_order)
    print(random_order[:10])

    split_idx = int(len(input_ids) * train_ratio)
    input_ids_train = np.array([input_ids[i] for i in random_order[:split_idx]])
    input_types_train = np.array([input_types[i] for i in random_order[:split_idx]])
    input_masks_train = np.array([input_masks[i] for i in random_order[:split_idx]])
    char_ids_train = np.array([char_ids[i] for i in random_order[:split_idx]])
    start_ids_train = np.array([start_ids[i] for i in random_order[:split_idx]])
    end_ids_train = np.array([end_ids[i] for i in random_order[:split_idx]])
    y_train = np.array([label[i] for i in random_order[:split_idx]])
    print("input_ids_train.shape:" + str(input_ids_train.shape))
    print("input_types_train.shape:" + str(input_types_train.shape))
    print("input_masks_train.shape:" + str(input_masks_train.shape))
    print("char_ids_train.shape:" + str(char_ids_train.shape))
    print("start_ids_train.shape:" + str(start_ids_train.shape))
    print("end_ids_train.shape:" + str(end_ids_train.shape))
    print("y_train.shape:" + str(y_train.shape))

    input_ids_test = np.array([input_ids[i] for i in random_order[split_idx:]])
    input_types_test = np.array([input_types[i] for i in random_order[split_idx:]])
    input_masks_test = np.array([input_masks[i] for i in random_order[split_idx:]])
    char_ids_test = np.array([char_ids[i] for i in random_order[split_idx:]])
    start_ids_test = np.array([start_ids[i] for i in random_order[split_idx:]])
    end_ids_test = np.array([end_ids[i] for i in random_order[split_idx:]])
    y_test = np.array([label[i] for i in random_order[split_idx:]])
    print("input_ids_test.shape:" + str(input_ids_test.shape))
    print("input_types_test.shape:" + str(input_types_test.shape))
    print("input_masks_test.shape:" + str(input_masks_test.shape))
    print("char_ids_test.shape:" + str(char_ids_test.shape))
    print("start_ids_test.shape:" + str(start_ids_test.shape))
    print("end_ids_test.shape:" + str(end_ids_test.shape))
    print("y_test.shape:" + str(y_test.shape))

    return (input_ids_train, input_types_train, input_masks_train, char_ids_train, start_ids_train, end_ids_train,
            y_train, input_ids_test, input_types_test, input_masks_test, char_ids_test, start_ids_test, end_ids_test,
            y_test)


def load_char_to_ids_dict(char_vocab_file):
    vocab = collections.OrderedDict()
    with open(char_vocab_file, "r", encoding="utf-8") as reader:
        chars = reader.readlines()
    for index, char in enumerate(chars):
        char = char.rstrip('\n')
        vocab[char] = index
    return vocab


def _get_char_to_ids_dict():
    global _CHAR_TO_IDS
    if _CHAR_TO_IDS is None:
        _CHAR_TO_IDS = load_char_to_ids_dict(char_vocab_file=CHARBERT_CHAR_VOCAB_FILE)
    return _CHAR_TO_IDS


def CharbertInput(context, tokenizer=None):
    tokenizer = tokenizer or _build_tokenizer()
    char2ids_dict = _get_char_to_ids_dict()
    max_length = 200
    char_maxlen = 200
    tokens = tokenizer.convert_ids_to_tokens(context)
    unk_id = char2ids_dict.get("[UNK]", 100)
    space_id = char2ids_dict.get(" ", unk_id)
    if space_id >= CHARBERT_CHAR_VOCAB_SIZE:
        space_id = unk_id

    char_ids = []
    start_ids = [char_maxlen - 1] * max_length
    end_ids = [char_maxlen - 1] * max_length
    char_cursor = 0

    for token_idx, token in enumerate(tokens[:max_length]):
        if token == "[PAD]":
            break

        token_text = token[2:] if token.startswith("##") else token
        if not token_text:
            continue

        token_start = char_cursor
        appended = 0
        for ch in token_text:
            if char_cursor >= char_maxlen:
                break

            cid = char2ids_dict.get(ch, unk_id)
            if cid >= CHARBERT_CHAR_VOCAB_SIZE:
                cid = unk_id
            char_ids.append(cid)
            char_cursor += 1
            appended += 1

        if appended > 0:
            start_ids[token_idx] = token_start
            end_ids[token_idx] = char_cursor - 1

        if token_idx < max_length - 1 and char_cursor < char_maxlen:
            char_ids.append(space_id)
            char_cursor += 1

    if len(char_ids) < char_maxlen:
        char_ids = char_ids + [0] * (char_maxlen - len(char_ids))
    else:
        char_ids = char_ids[:char_maxlen]

    return char_ids, start_ids, end_ids

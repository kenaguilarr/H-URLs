# Malicious URL Detection - PMANet

This repository contains a Python/PyTorch implementation for malicious URL detection using PMANet-style BERT/CharBERT encoders. The current codebase supports three main workflows:

1. URL-only detection using `url_Train.py` and `url_Test_binary.py`
2. HTML-only detection using `html_Train.py` and `html_Test_binary.py`
3. URL + HTML fusion detection using `fusion_train.py` and `fusionTest_binary.py`

The project is based on the paper [Malicious URL Detection via Pretrained Language Model-Guided Multi-Level Feature Attention Network](https://arxiv.org/abs/2311.12372), with additional repository scripts for HTML and fusion experiments.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `Data/` | Datasets used for URL, HTML, adversarial, cross-data, and experiment runs. |
| `Data/Raw_Dataset_QR/` | Main raw dataset used by the URL, HTML, and fusion scripts. |
| `charbert-bert-wiki/charbert-bert-wiki/` | Required pretrained CharBERT files: `config.json`, `pytorch_model.bin`, and `vocab.txt`. |
| `Model_PMA.py` | Main model definitions for URL CharBERT and HTML BERT-style branches. |
| `Model_CharBERT.py` | CharBERT model implementation. |
| `attention.py` | Channel attention layer used by the PMANet model. |
| `url_dataprocessing.py` | URL preprocessing and CharBERT input preparation. |
| `html_dataprocessing.py` | HTML folder/file preprocessing. |
| `fusion_dataprocessing.py` | Builds paired URL + HTML samples for fusion training/testing. |
| `url_Train.py` | Trains the URL-only CharBERT model and saves `urlmodel_charbert.pth`. |
| `html_Train.py` | Trains the HTML-only model and saves `htmlmodel.pth`. |
| `fusion_train.py` | Trains the fusion model and saves `fusion_model.pth`. |
| `url_Test_binary.py` | Tests the URL-only checkpoint. |
| `html_Test_binary.py` | Tests the HTML-only checkpoint. |
| `fusionTest_binary.py` | Tests the fusion checkpoint. |
| `Experiment results/` and `results_figures/` | Stored plots, metrics, and experiment artifacts. |
| `PREPROCESSING_EXPLANATION_GUIDE.md` | More detailed explanation of preprocessing behavior. |

## Requirements

Recommended environment:

- Python 3.8
- PyTorch
- CUDA-capable GPU recommended for training

Install the main Python packages:

```bash
pip install torch transformers pytorch-pretrained-bert pandas numpy scikit-learn matplotlib seaborn tqdm openpyxl
```

Optional packages used by adversarial/data utility scripts:

```bash
pip install tld tensorflow
```

## Step 1: Prepare the Repository

Open a terminal in the repository root:

```bash
cd "C:\Users\jkena\OneDrive\Desktop\Malicious-URL-Detection-PMANet(1)"
```

Create and activate a virtual environment if needed:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```bash
pip install torch transformers pytorch-pretrained-bert pandas numpy scikit-learn matplotlib seaborn tqdm openpyxl
```

## Step 2: Check the Pretrained CharBERT Files

The model code expects the pretrained CharBERT files at:

```text
charbert-bert-wiki/charbert-bert-wiki/
```

Make sure this folder contains:

```text
config.json
pytorch_model.bin
vocab.txt
```

These files are required by `Model_PMA.py`, `url_dataprocessing.py`, and `fusion_dataprocessing.py`.

## Step 3: Check the Dataset Layout

The default scripts use hard-coded paths under:

```text
Data/Raw_Dataset_QR/
```

Expected URL training files:

```text
Data/Raw_Dataset_QR/url/url_train/Train_Benign.xlsx
Data/Raw_Dataset_QR/url/url_train/Train_Malicious.xlsx
```

Expected HTML training folders:

```text
Data/Raw_Dataset_QR/html/html_benign_train/
Data/Raw_Dataset_QR/html/html_malicious_train/
```

Expected cross-data URL testing file:

```text
Data/Raw_Dataset_QR/CrossData/url/CD_url _testbinary.csv
```

Expected cross-data HTML testing folders:

```text
Data/Raw_Dataset_QR/CrossData/html/Mendeley_benign/
Data/Raw_Dataset_QR/CrossData/html/Mendeley_malicious/
```

If your data is stored somewhere else, update the paths inside the corresponding training or testing script before running.

## Step 4: Train the URL-Only Model

Run:

```bash
python url_Train.py
```

What happens:

1. Reads benign and malicious URL Excel files.
2. Converts URLs into BERT/CharBERT tensors.
3. Splits data into training and validation sets using `train_ratio=0.95`.
4. Trains for `num_epochs=3`.
5. Saves the best checkpoint as:

```text
urlmodel_charbert.pth
```

This checkpoint is also required by the fusion model.

## Step 5: Train the HTML-Only Model

Run:

```bash
python html_Train.py
```

What happens:

1. Reads HTML/text files from the benign and malicious training folders.
2. Converts file contents into BERT-style input tensors.
3. Splits data into training and validation sets using `train_ratio=0.95`.
4. Trains for `NUM_EPOCHS=3`.
5. Saves the best checkpoint as:

```text
htmlmodel.pth
```

This checkpoint is also required by the fusion model.

## Step 6: Train the Fusion Model

Train fusion after `urlmodel_charbert.pth` and `htmlmodel.pth` already exist.

Run the default training:

```bash
python fusion_train.py
```

For a quick smoke test with fewer samples:

```bash
python fusion_train.py --max-train-samples 20 --max-val-samples 10 --num-epochs 1
```

Common options:

```bash
python fusion_train.py --batch-size 4 --num-epochs 3 --train-ratio 0.95 --checkpoint-path fusion_model.pth
```

What happens:

1. Loads paired URL + HTML samples from `Data/Raw_Dataset_QR`.
2. Encodes URL inputs with the CharBERT URL branch.
3. Encodes HTML inputs with the HTML branch.
4. Loads the pretrained URL and HTML checkpoints.
5. Freezes both encoders by default.
6. Trains the fusion classifier head.
7. Saves the best checkpoint as:

```text
fusion_model.pth
```

## Step 7: Test the URL-Only Model

Run:

```bash
python url_Test_binary.py
```

Outputs:

```text
confusion_matrix.png
roc_curve.png
results.txt
```

The script loads:

```text
urlmodel_charbert.pth
Data/Raw_Dataset_QR/CrossData/url/CD_url _testbinary.csv
```

## Step 8: Test the HTML-Only Model

Run:

```bash
python html_Test_binary.py
```

Outputs:

```text
confusion_matrix.png
html_roc_curve.png
results.txt
```

The script loads:

```text
htmlmodel.pth
Data/Raw_Dataset_QR/CrossData/html/Mendeley_benign/
Data/Raw_Dataset_QR/CrossData/html/Mendeley_malicious/
```

## Step 9: Test the Fusion Model

Run:

```bash
python fusionTest_binary.py
```

For a quick test:

```bash
python fusionTest_binary.py --max-samples 20
```

Common options:

```bash
python fusionTest_binary.py --checkpoint-path fusion_model.pth --batch-size 1
```

Outputs:

```text
fusion_confusion_matrix.png
fusion_roc_curve.png
fusion_results.txt
```

The script loads:

```text
urlmodel_charbert.pth
htmlmodel.pth
fusion_model.pth
```

## Step 10: Read the Results

Training and testing scripts print metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Average loss

The testing scripts also save:

- Confusion matrix image
- ROC curve image
- Per-sample prediction results in `.txt` format

Previously generated experiment files are stored in:

```text
Experiment results/
results_figures/
```

## Typical Workflow

Run the repository in this order:

```bash
python url_Train.py
python html_Train.py
python fusion_train.py
python url_Test_binary.py
python html_Test_binary.py
python fusionTest_binary.py
```

For a faster first check:

```bash
python fusion_train.py --max-train-samples 20 --max-val-samples 10 --num-epochs 1
python fusionTest_binary.py --max-samples 20
```

## Important Notes

- Many scripts use hard-coded dataset paths. If a file is missing, check the path in the script first.
- Training can be slow on CPU because the models are BERT/CharBERT based.
- The fusion model expects existing URL and HTML checkpoints before training or testing.
- Several scripts write output files with the same names, such as `confusion_matrix.png` and `results.txt`. Move or rename outputs if you want to preserve multiple runs.
- The folder `appendix_code/` contains source code copies used for appendix/report generation, not the main runtime entry points.

## Troubleshooting

### `FileNotFoundError` for CharBERT files

Check that this folder exists:

```text
charbert-bert-wiki/charbert-bert-wiki/
```

and contains `config.json`, `pytorch_model.bin`, and `vocab.txt`.

### Excel loading error

Install `openpyxl`:

```bash
pip install openpyxl
```

### CUDA out-of-memory error

Lower the batch size in the script or command:

```bash
python fusion_train.py --batch-size 1
```

For `url_Train.py` and `html_Train.py`, edit the `batch_size` or `BATCH_SIZE` variable inside the file.

### Fusion checkpoint error

Make sure these files exist before running fusion:

```text
urlmodel_charbert.pth
htmlmodel.pth
fusion_model.pth
```

Use `fusion_train.py` to create `fusion_model.pth` if it does not exist yet.

## License

See `LICENSE`.

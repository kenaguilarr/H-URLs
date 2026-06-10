import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from html_dataprocessing import dataPreprocessFromHTMLFolder, spiltDatast_bert
from Model_PMA import Model


def train(model, device, train_loader, optimizer, epoch):                   

    model.train()
    
    for batch_idx, (x1, x2, x3, y) in enumerate(train_loader):
        optimizer.zero_grad()

        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        y = y.to(device).long().view(-1)

        outputs, pooled, logits = model([x1, x2, x3])

        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
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

              
    test_loss /= len(test_loader)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

                      
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['benign', 'malware'],
                yticklabels=['benign', 'malware'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    print(
        'Test set: Average loss: {:.4f}, Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%, F1: {:.2f}%'
        .format(test_loss, accuracy * 100, precision * 100, recall * 100, f1 * 100)
    )

    return accuracy, precision, recall, f1



def main():
    input_ids = []                  
    input_types = []               
    input_masks = []                  
    label = []
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    benign_dir = "Data/Raw_Dataset_QR/html/html_benign_train"
    malicious_dir = "Data/Raw_Dataset_QR/html/html_malicious_train"

    dataPreprocessFromHTMLFolder(benign_dir, input_ids, input_types, input_masks, label, urltype=0, max_chars=50000)
    dataPreprocessFromHTMLFolder(malicious_dir, input_ids, input_types, input_masks, label, urltype=1, max_chars=50000)


             
    input_ids_train, input_types_train, input_masks_train, y_train,\
    input_ids_val, input_types_val, input_masks_val, y_val = spiltDatast_bert(
        input_ids, input_types, input_masks, label, train_ratio=0.95
    )

                                                                        
    BATCH_SIZE = 4

    train_data = TensorDataset(
        torch.tensor(input_ids_train, dtype=torch.long),
        torch.tensor(input_types_train, dtype=torch.long),
        torch.tensor(input_masks_train, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.long).view(-1)
    )
    train_loader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE)

    val_data = TensorDataset(
        torch.tensor(input_ids_val, dtype=torch.long),
        torch.tensor(input_types_val, dtype=torch.long),
        torch.tensor(input_masks_val, dtype=torch.long),
        torch.tensor(y_val, dtype=torch.long).view(-1)
    )
    val_loader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=BATCH_SIZE)

    model = Model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    best_acc = 0.0
    NUM_EPOCHS = 3
    PATH = 'htmlmodel.pth'
    for epoch in range(1, NUM_EPOCHS + 1):            
        train(model, DEVICE, train_loader, optimizer, epoch)
        acc, precision, recall, f1 = validation(model, DEVICE, val_loader)

        if best_acc < acc:
            best_acc = acc
            torch.save(model.state_dict(), PATH)                       
        print("acc is: {:.4f}, best acc is {:.4f}\n".format(acc, best_acc))

if __name__ == '__main__':
    main()

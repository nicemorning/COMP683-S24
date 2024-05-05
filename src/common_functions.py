import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import DataLoader
import time

def test_model(model, test_ld, device):
    print(f'TestMode')
    model.eval()
    correct = 0
    total = 0
    t0 = time.time()
    test_predicted = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_ld):
            inputs, targets = inputs.to(device), targets.to(device)
            x = inputs           
            outputs = model(x)
            _, predicted = outputs.max(1)
            test_predicted.extend(predicted.cpu().numpy())
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    t1 = time.time()        
    acc = 100.0 * correct / total
    print(f'Acc:{acc:.2f}% Time:{(t1-t0):.3f}')
    return test_predicted, acc

def train_model(model, optimizer, criterion, train_ld, epoch, device):
    print(f'TrainMode')
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    t0 = time.time()   
    for batch_idx, (inputs, targets) in enumerate(train_ld):
        inputs, targets = inputs.to(device), targets.to(device)
        x = inputs
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        # Calculate accuracy
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    t1 = time.time()
    acc = 100.0 * correct / total
    print(f'correct:{correct} total:{total}')
    avg_batch_loss = train_loss /(batch_idx+1) 
    print(f'Train Iter:{epoch} Loss:{avg_batch_loss:.6f} Acc:{acc:.4f} Time:{(t1-t0):.3f}')
    return avg_batch_loss, acc 
    
    
def filter_by_class(X, Y, classes):
    mask = torch.zeros_like(Y, dtype=torch.bool)
    for cls in classes:
        mask |= (Y == cls)
    return X[mask], Y[mask]

def remap_labels_to_top_classes(labels, top_classes):
    # Map old class indices to new class indices based on their position in top_classes
    class_mapping = {old_class: new_class for new_class, old_class in enumerate(top_classes.tolist())}
    
    # Remap labels using the class mapping
    remapped_labels = torch.tensor([class_mapping[label.item()] if label.item() in class_mapping else label for label in labels])
    return remapped_labels

def top_k_data(X_train_ts, Y_train_ts, X_test_ts, Y_test_ts, k):
    if k == 33:
        print('Use full dataset!')
        trainset = TensorDataset(X_train_ts, Y_train_ts)
        testset = TensorDataset(X_test_ts, Y_test_ts)
    else:
        # Step 1: Count the number of samples per class in the training data
        class_counts = torch.bincount(Y_train_ts)

        # Step 2: Get the indices of the top k most frequent classes
        _, top_classes = torch.topk(class_counts, k)

        # Filter the datasets
        X_train_filtered, y_train_filtered = filter_by_class(X_train_ts, Y_train_ts, top_classes)
        X_test_filtered, y_test_filtered = filter_by_class(X_test_ts, Y_test_ts, top_classes)

        # Remap labels directly using the mapping defined in remap_labels_to_top_classes
        y_train_filtered = remap_labels_to_top_classes(y_train_filtered, top_classes)
        y_test_filtered = remap_labels_to_top_classes(y_test_filtered, top_classes)
        print(X_train_filtered.shape, y_train_filtered.shape, X_test_filtered.shape, y_test_filtered.shape)
        # Create TensorDatasets
        trainset = TensorDataset(X_train_filtered, y_train_filtered)
        testset = TensorDataset(X_test_filtered, y_test_filtered)
    return trainset, testset


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

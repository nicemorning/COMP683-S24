import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tcga_files import tcga_labels
from common_functions import  train_model, test_model, CustomDataset
import time
import sys
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

num_classes = 33

data = torch.load('feat2img.bin')
X_train_tensor = data['x_train']
y_train_tensor = data['y_train']
X_test_tensor = data['x_test']
y_test_tensor = data['y_test']
print(X_train_tensor.shape, y_train_tensor.shape, X_test_tensor.shape, y_test_tensor.shape)

plot_labels = list(tcga_labels.keys())
print(plot_labels)

# Define your transformations
transform = transforms.Compose([
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),  # Flips the image horizontally with p=0.5 (default)
])

criterion = nn.CrossEntropyLoss()


def run_test(model_name, train_batch_size, pre_trained=True):
    X_train_new = X_train_tensor
    y_train_new = y_train_tensor
    
    X_test_new = X_test_tensor
    y_test_new = y_test_tensor
    
    trainset = CustomDataset(X_train_new, y_train_new, transform=None)
    print(len(trainset))
    trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=4)
    print(len(trainloader))
    
    testset = CustomDataset(X_test_new, y_test_new)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=pre_trained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 33)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pre_trained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 33)
    elif model_name == 'mobilenet':
        model = torchvision.models.mobilenet_v2(pretrained=pre_trained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, 33)
    elif model_name == 'squeezenet1':
        model = torchvision.models.squeezenet1_1(pretrained=pre_trained)
        final_conv = torch.nn.Conv2d(512, 33, kernel_size=(1,1), stride=(1,1))
        model.classifier[1] = final_conv
        model.num_classes = 33    
    else:
        print(f'ERROR {model_name}')
        sys.exit()
    
    
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of parameters: ", total_params)
    model.to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    test_acc_list = []
    train_acc_list = []
    max_test_acc = 0
    best_model_fname = ''
    
    for i in range(18):
        train_loss, train_acc = train_model(model, optimizer, criterion, trainloader, i, device)
        test_predicted, test_acc = test_model(model, testloader, device)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
         
        if i == 4:
            optimizer.param_groups[0]['lr'] = 0.005
            print(optimizer)

        if i == 8:
            optimizer.param_groups[0]['lr'] = 0.0025
            print(optimizer)           
                
        if i == 12:
            optimizer.param_groups[0]['lr'] = 0.001
            print(optimizer)
            
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            model_fname= f'{model_name}_b{train_batch_size}_{i}.pth'
            print(model_fname)
            if best_model_fname:
                os.remove(best_model_fname)  # Remove the previous best model
            best_model_fname = model_fname  # Update best model
            torch.save(model.state_dict(), model_fname)
        
    print(f'*****************Best Test ACC:{model_name}_{train_batch_size}:{max_test_acc:.4f}*************************')
    # Save accuracy lists to a text file
    with open('test_acc.txt', 'a') as f:
        f.write(f'test_acc_{model_name}_b{train_batch_size} = {test_acc_list}\n')
        f.write(f'train_acc_{model_name}_b{train_batch_size} = {train_acc_list}\n')
        

pretrain = True
for train_batch_size in [64, 40, 32, 20, 16]:
    run_test('resnet18', train_batch_size, pretrain)


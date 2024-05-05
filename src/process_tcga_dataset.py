import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE 
from tcga_files import files_count, files_fpkm, files_fpkm_uq, tcga_labels
from image_transformer import ImageTransformer
from utilities import Norm2Scaler
import gc
import os 

# Function to process images in batches
def process_in_batches(image_list, batch_size):
    batched_tensors = []
    for i in range(0, len(image_list), batch_size):
        batch = image_list[i:i + batch_size]
        tensors = [preprocess(img) for img in batch]
        batched_tensors.append(torch.stack(tensors).float())
    return torch.cat(batched_tensors, dim=0)
    
all_data = []
for i, file in enumerate(files_fpkm_uq, start=1):
    label_name = file.split('-')[1].split('.')[0]
    print(label_name)
    label = tcga_labels[label_name]
    print(label)
      
    # Read data
    # Make the path OS independent 
    file_name = os.path.join('TCGA', file)
    print(file_name)
    data = pd.read_csv(file_name, sep='\t', index_col=0)
    print(data.shape)
    #print(data.head())
    
    # Append data and labels to all_data
    for column in data.columns:
        all_data.append((data[column].values, label))
    
print(f'{i} files have been processed')
# Separate features and labels
features = [sample[0] for sample in all_data]
labels = [sample[1] for sample in all_data]

X = np.array(features,dtype=np.float32)
y = np.array(labels,dtype=np.int8)
# print(X[:5,:5].T, y[:5])
# print(X.shape, y.shape)
# print(X.dtype, y.dtype)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y)

# Delete unused variables to save memory since they are no longer needed after splitting
del data, all_data, features, labels, X, y
gc.collect()

# Convert features to 2-D images 
ln = Norm2Scaler()
X_train_norm = ln.fit_transform(X_train)
X_test_norm = ln.transform(X_test)

# Delete X_train, X_test to free up memory
del X_train, X_test
gc.collect()

distance_metric = 'cosine'
reducer = TSNE(
    n_components=2,
    metric=distance_metric,
    init='random',
    learning_rate='auto',
    n_jobs=-1
)

pixel_size = (227,227)
img_trans = ImageTransformer(feature_extractor=reducer, pixels=pixel_size)

img_trans.fit(X_train_norm, y=y_train, plot=False)

X_train_img = img_trans.transform(X_train_norm)
X_test_img = img_trans.transform(X_test_norm)

del X_train_norm, X_test_norm
gc.collect()

preprocess = transforms.Compose([
    transforms.ToTensor()
])

X_train_tensor = process_in_batches(X_train_img, batch_size=100)
#X_train_tensor = torch.stack([preprocess(img) for img in X_train_img]).float()
y_train_tensor = torch.from_numpy(y_train)

del X_train_img, y_train
gc.collect()

X_test_tensor = process_in_batches(X_test_img, batch_size=100) 
#X_test_tensor = torch.stack([preprocess(img) for img in X_test_img]).float()
y_test_tensor = torch.from_numpy(y_test)

del X_test_img, y_test
gc.collect()

print(X_train_tensor.shape, y_train_tensor.shape)

torch.save({'x_train':X_train_tensor, 'y_train':y_train_tensor, 'x_test':X_test_tensor, 'y_test':y_test_tensor}, 'feat2img.bin')

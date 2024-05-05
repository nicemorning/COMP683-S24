## Introduction
This repository hosts the COMP683 final project: Tumor Type Prediction using Convolutional Neural Networks (CNNs). The classification results for 33 tumor types are summarized in the following table.

| CNN Architecture | # Parameters | Test Accuracy (%) |
|------------------|--------------|-------------------|
| ResNet18         | 11,193,441   | 95.30             |
| ResNet50         | 23,575,649   | 94.85             |
| MobileNetV2      | 2,266,145    | 94.30             |
| SqueezeNet      | 739,425      | 91.14             |

## Repository Structure

- **src/**: Contains all source files.
  - **common_functions.py**: Includes common functions for CNN training and testing.
  - **image_transformer.py**: Transforms 1-D features to 2-D images using feature similarity, adapted from DeepInsight.
  - **process_tcga_dataset.py**: Reads TCGA RNA-seq data for 33 tumor types and converts the sample-feature matrix into 2-D images.
  - **test_cnns.py**: Trains and evaluates prediction accuracy across 33 tumor types using various CNN architectures on converted 2-D images.
  - **tcga_files.py**: Lists predefined filenames and tumor names for the 33 tumor types, sourced from the UCSC Xena Portal.

## How to Reproduce the Results
### 1. Download TCGA RNA-Seq Data

- Visit the [UCSC Xena Portal](https://xenabrowser.net/datapages/).
- Select the Harmonized dataset by clicking on the **GDC hub** checkbox.
- Download the required files listed in `./src/TCGA/tcga_file_list.txt` into the `./src/TCGA` folder

### 2. Convert Gene Expression Data to 2-D Images

Navigate to the `src` directory, run the following Python script to convert the TCGA gene expression data into 2-D images:

`python process_tcga_dataset.py`

This script will generate a file named feat2img.bin, which includes all the 2-D images along with their corresponding labels.

### 3. Train and Evaluate CNN Models
Continue in the `src` directory, execute following Python script to train and evaluate the prediction accuracy for 33 tumor types using various CNN architectures, including SqueezeNet, MobileNet, ResNet18, and ResNet50:

`python test_cnns.py`

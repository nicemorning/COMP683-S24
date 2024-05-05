### 1. Download TCGA RNA-Seq Data

- Visit the [UCSC Xena Portal](https://xenabrowser.net/datapages/).
- Select the Harmonized dataset by clicking on the **GDC hub** checkbox.
- Download the required files listed in `./src/TCGA/tcga_file_list.txt` to the `./src/TCGA` folder

### 2. Convert Gene Expression Data to 2-D Images

Run the following Python script to convert the TCGA gene expression data into 2-D images:

`cd src`

`python process_tcga_dataset.py`

This script will generate a file named feat2img.bin, which includes all the 2-D images along with their corresponding labels.

### 3. Train and Evaluate CNN Models:
Execute following Python script to train and evaluate the prediction accuracy for 33 tumor types using various CNN architectures, including SqueezeNet, MobileNet, ResNet18, and ResNet50:

`cd src`

`python test_cnns.py`

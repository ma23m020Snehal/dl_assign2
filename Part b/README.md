# Fine-Tuning a Pre-Trained Model on iNaturalist Dataset

 I have created this notebook in kaggle.

This notebook shows **fine-tuning a pre-trained model (ResNet50)** on the **iNaturalist dataset** using **PyTorch**. 

The workflow includes setting up the environment, loading data, modifying architecture, training, validation, and logging experiments with **Weights & Biases (wandb)**.

## Objective

To use transfer learning by fine-tuning a model pre-trained on ImageNet to classify images from the iNaturalist dataset, using the best hyperparameters obtained from the sweeps.


## Requirements

- Python ≥ 3.7
- PyTorch
- torchvision
- wandb
- matplotlib
- numpy
- pandas

Install dependencies:

```bash
pip install torch torchvision wandb matplotlib pandas
```

---

##  Usage

Run the notebook using `Kaggle` or any Jupyter environment.

1. **Initialize Weights & Biases:**

   You'll be prompted to enter your W&B API key:
   ```python
   import wandb
   wandb.login()
   ```

2. **Data Preparation:**

   - Uses the iNaturalist 12K dataset
   - The training data is split into **80% training** and **20% validation**

3. **Model Configuration:**
    setting up a pre-trained model from the torchvision.models library and preparing it for fine-tuning on the iNaturalist dataset. The core idea is to transfer knowledge from models trained on ImageNet to a new task, saving compute and improving performance on small datasets. Model is loaded with **pretrained=True**, so it comes with weights learned from the ImageNet dataset.

    Freeze percentage of layers from [0.2, 0.4, 0.6, 0.8, 0.9, 1.0] is set through sweep configuration.

    Once the base model is loaded and frozen according to your strategy, the final classification layer is modified to match the number of output classes in the iNaturalist dataset.


4. **Training Loop:**

   - Runs for specified number of epochs
   - Monitors training and validation loss/accuracy
   - Logs metrics and images to Weights & Biases


## Sample Output

- **Validation Accuracy: 78.58929 (maximum)**
- **W&B Dashboard: https://wandb.ai/snehalma23m020-iit-madras/fine-tune-inaturalist/reports/MA23M020-Snehal-DA6401-Assignment-2--VmlldzoxMjMzMTYxNA?accessToken=umsyi0exaqkod1g6hzzuy52l66bvxmbwicw7egg9fomhr076aztl0m4teyel6aal** 
-**Github repository link : https://github.com/ma23m020Snehal/dl_assign2/tree/master/Part%20b**

## Wandb sweep:
    {method: bayes
    metric: Validation accuracy
    goal: Maximize }
Maximum validation accuracy: 78.58929 %  over 31 epochs

## code organisation 
```
/Part b/
├── README.md                    # Description and hyperparameter info for Part A
├── part-b-latest-fine-tuning-a-pre-trained-model.ipynb    # experiments
├── partb-test.ipynb              # Notebook for final evaluation and visualization
└── trainb.py                     # Main training script using CLI argparse
```

## trainb.py file 
fine-tuned a pretrained ResNet50 model on the iNaturalist dataset using PyTorch, with support for configurable hyperparameters via command-line arguments. It logs training, validation, and test metrics to Weights & Biases (W&B) and supports data augmentation, layer freezing, and stratified train-validation splitting.

# To run trainb.py :

with default values :
```python train.py --wandb_snehalma23m020-iit-madras```

to override the values of hyperparamters
```python train.py --lr 0.0002 --epochs 20 --data_aug no --wandb_project fine-tune-inaturalist --wandb_snehalma23m020-iit-madras```


# best hyperparamaters obtained from this are :
- batch size=64
- data_augmentation = no 
- epochs = 15
- freeze ratio = 0.8
- l2_reg = 0.0001
- model=resnet 50
 

# test accuracy obtained :
Test Accuracy: 0.7770

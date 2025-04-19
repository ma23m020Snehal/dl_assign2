#  DL Assignment 2 — iNaturalist Image Classification  
**Course:** DA6401 - Deep Learning  
**Student:** Snehal (MA23M020), IIT Madras

 **Assignment Goal:**  
Train a CNN from scratch in part a and fine tune pretrained model in partb of assignment  on the iNaturalist 12K dataset and optimize its performance through hyperparameter sweeps using Weights & Biases (W&B).

---

##  Links

-  [W&B Report Dashboard](https://wandb.ai/snehalma23m020-iit-madras/fine-tune-inaturalist/reports/MA23M020-Snehal-DA6401-Assignment-2--VmlldzoxMjMzMTYxNA?accessToken=umsyi0exaqkod1g6hzzuy52l66bvxmbwicw7egg9fomhr076aztl0m4teyel6aal)  
-  [GitHub Repository](https://github.com/ma23m020Snehal/dl_assign2)

---

##  Code Organization
             
```
ma23m020snehal-dl_assign2/
    ├── README.md                         # Overall project summary with links and structure
    ├── Part a/
    │   ├── README.md                     # Description and hyperparameter info for Part A
    │   ├── parta-test.ipynb              # Notebook for final evaluation and visualization
    │   ├── parta-train.ipynb             # experiments
    │   └── traina.py                     # Main training script using CLI argparse
    └── Part b/
        ├── README.md                    # Description and hyperparameter info for Part A
        ├── part-b-latest-fine-tuning-a-pre-trained-model.ipynb    # experiments
        ├── partb-test.ipynb              # Notebook for final evaluation and visualization
        └── trainb.py                     # Main training script using CLI argparse
```





##  Dependencies

Make sure the following libraries are installed:

```bash
pip install torch torchvision wandb matplotlib tqdm
```

---


##  Dataset Structure

```
/inaturalist_12K/
├── train/         # Used for 80% train and 20% validation
├── val/           # Held-out test set for final evaluation
```

---



##  References


- [building-a-convolutional-neural-network-in-pytorch](https://machinelearningmastery.com/building-a-convolutional-neural-network-in-pytorch/ )
- [Ultimate Guide to Fine-Tuning in PyTorch ](https://rumn.medium.com/part-1-ultimate-guide-to-fine-tuning-in-pytorch-pre-trained-model-and-its-configuration-8990194b71e)


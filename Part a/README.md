# Part A - CNN Training from Scratch on iNaturalist Dataset

1. Building a CNN model with 5 convolutional blocks followed by a dense classifier.
2. Ensuring configurability of filter count, activation, dropout, etc.
3. Performing stratified 80/20 train-validation split.
4. Using Weights & Biases (W&B) for hyperparameter sweeps.
5. Evaluating performance on iNaturalist dataset.


## Model Architecture

- The model (`AdaptiveCNN`) contains:
  - **5 convolutional blocks**, each consisting of:
    - `Conv2D` layer
    - Configurable **activation function** (`ReLU`, `GELU`, `SiLU`, `Mish`)
    - Optional **BatchNorm**
    - **MaxPool2D**
    - **Dropout**
  - One **fully connected (dense)** layer
  - Final **output layer** with **10 neurons** (for 10 iNaturalist classes)

### Configurable Parameters:
- Number of filters: `num_filters` (same or variable across layers) ; "values"= [32, 64, 128] 
- Filter organization: `same`, `double`, `half` ;  values = ["same", "double", "half"]
- Kernel sizes: `kernel_size`
- Activation: `act_fn` ; "values"= ["relu", "gelu", "silu", "mish", "tanh"] 
- Batch normalization: `Yes/No` ;  "values" = [True, False]
- Dropout rate: `dropout_rate` ;  "values" = [0, 0.2, 0.3, 0.5]
- Number of neurons in dense layer: `num_neurons` ;  "values" = [64, 128, 256] 
- Data Augmentation : `data_augmentation` ; values" = [True, False]
- Learning Rate : `learning_rate` ; "values" = [0.0005, 0.001, 0.0001] 
- Batch Size : `batch_size`; "values" = [32, 64]
- Kernel Size : `kernel_size` ; "values" = [[3]*5, [3, 5, 5, 7, 7], [5]*5, [7]*5, [7, 5, 5, 3, 3]]
- L2 regularisation : `l2_reg` ; "values" = [0, 0.0005, 0.05]
- Epochs : `epochs` ; "values" = [15,17,20]


##  Dataset Structure

```
/inaturalist_12K/
│
├── train/   ← Used for both training + validation
│
├── val/     ← Used only as the test set (never used during tuning)
```

---

## Library Dependencies

- Python ≥ 3.8
- PyTorch
- torchvision
- wandb
- matplotlib
- Kaggle environment (optional)


##  Running the Code

### 1. Install Weights & Biases and login:
```python
!pip install wandb
import wandb
wandb.login(key="YOUR_WANDB_API_KEY")
```

### 2. Start a sweep:
```python
sweep_id = wandb.sweep(sweep_config, project="q2_assign2_exp7")
wandb.agent(sweep_id, function=train, count=30)
```


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

# traina.py file
this can be run with the help of command below
```python train.py --wandb_project DL_assignment_2_eval --wandb_snehalma23m020-iit-madras```


# Testing
- After identifying the best config, the model is retrained and evaluated on:
  - Training Set
  - Validation Set
  - Test Set (`val/`)

- Predictions are visualized using a **10×3 grid** of sample images and their predicted labels
   Each row corresponds to one class, Each column shows a sample image with the predicted label ,The visualization is logged to W&B via `wandb.Image(fig)`

#  Best hyperparametrs obtained :
- num_filters =	64
- filter_org = double
- act_fn = GELU
- dropout_rate = 0.3
- batch_norm = True
- data_augmentation = True
- num_neurons = 256
- learning_rate = 0.001
- batch_size = 64
- kernel_size = [3, 5, 5, 7, 7]
- l2_reg = 0.0005
- epochs = 20

# Test accuracy obtained : 34.90%
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import random
import wandb
import argparse
import matplotlib.pyplot as plt
import numpy as np


# CNN Model Class
def compute_img_size(img_w, filter_size, padding, stride):
    return (1 + (img_w - filter_size + (2 * padding)) / stride)

class AdaptiveCNN(nn.Module):
    def __init__(self, input_size=224, input_channels=3, **config):
        super(AdaptiveCNN, self).__init__()
        filters = [config['num_filters']] * 5
        if config['filter_org'] == 'double':
            filters = [config['num_filters'] * (2 ** i) for i in range(5)]
        elif config['filter_org'] == 'half':
            filters = [max(config['num_filters'] // (2 ** i), 1) for i in range(5)]

        act_func = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'mish': nn.Mish()
        }
        activation = act_func[config['act_fn']]
        padding_vals = [k // 2 for k in config['kernel_size']]

        layers = []
        in_channels = input_channels
        for i in range(5):
            layers.append(nn.Conv2d(in_channels, filters[i], kernel_size=config['kernel_size'][i], padding=padding_vals[i]))
            layers.append(nn.BatchNorm2d(filters[i]) if config['batch_norm'] else nn.Identity())
            layers.append(activation)
            layers.append(nn.Dropout(p=config['dropout_rate']))
            layers.append(nn.MaxPool2d(2))
            in_channels = filters[i]

        self.conv_layers = nn.Sequential(*layers)

        size = input_size
        for k, p in zip(config['kernel_size'], padding_vals):
            size = compute_img_size(size, k, p, 1) // 2

        self.flattened_size = int(size * size * filters[-1])
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, config['num_neurons']),
            activation,
            nn.Dropout(p=config['dropout_rate']),
            nn.Linear(config['num_neurons'], 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Data Preparation Functions

def stratified_split(dataset, val_fraction=0.2, seed=42):
    random.seed(seed)
    label_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_indices[label].append(idx)

    train_indices, val_indices = [], []
    for label, indices in label_to_indices.items():
        n_val = int(len(indices) * val_fraction)
        random.shuffle(indices)
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
    return Subset(dataset, train_indices), Subset(dataset, val_indices)

def data_load(data_dir, use_aug):
    base_transforms = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    augment = [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    transform = transforms.Compose(augment + base_transforms) if use_aug else transforms.Compose(base_transforms)
    full_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform)
    return stratified_split(full_dataset)

def test_data_load(data_dir, use_aug):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)

# Visualize Predictions
def img_plot(model, test_loader):
    classes = test_loader.dataset.classes
    class_images = {cls: [] for cls in range(10)}
    for x, y in test_loader:
        for img, label in zip(x, y):
            if len(class_images[label.item()]) < 3:
                class_images[label.item()].append(img)
        if all(len(imgs) == 3 for imgs in class_images.values()):
            break

    model.eval()
    fig, axes = plt.subplots(10, 3, figsize=(12, 20))
    for i in range(10):
        for j in range(3):
            img = class_images[i][j].unsqueeze(0).cuda()
            with torch.no_grad():
                pred = torch.argmax(model(img), dim=1).item()
            npimg = img.squeeze(0).cpu().permute(1, 2, 0).numpy()
            axes[i][j].imshow((npimg * 0.5 + 0.5).clip(0, 1))
            axes[i][j].axis('off')
            axes[i][j].set_title(f"True: {classes[i]}\nPred: {classes[pred]}")
    plt.tight_layout()
    wandb.log({"10x3 Predictions Grid": wandb.Image(fig)})
    plt.show()


# Training, Validation & Testing
def model_train_val_and_test(model, train_data, val_data, test_data, config):
    wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=vars(config))
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.l2_reg)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

    for epoch in range(config.epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_correct, val_total, val_loss_total = 0, 0, 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss_total += loss.item() * x.size(0)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)

        val_loss = val_loss_total / val_total
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{config.epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

    # Final evaluation on test set
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_correct += (out.argmax(1) == y).sum().item()
            test_total += y.size(0)
    test_acc = test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")
    wandb.log({'test_accuracy': test_acc})
    img_plot(model, test_loader)

# Argument Parser for Best Config

def parse_args():
    parser = argparse.ArgumentParser(description="Train Adaptive CNN on iNaturalist")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_filters', type=int, default=32)
    parser.add_argument('--filter_org', type=str, default='same', choices=['same', 'double', 'half'])
    parser.add_argument('--kernel_size', nargs='+', type=int, default=[3, 3, 3, 3, 3])
    parser.add_argument('--act_fn', type=str, default='relu', choices=['relu', 'tanh', 'gelu', 'silu', 'mish'])
    parser.add_argument('--num_neurons', type=int, default=256)
    parser.add_argument('--batch_norm', type=bool, default=True)
    parser.add_argument('--dropout_rate', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--l2_reg', type=float, default=0.0005)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--data_augmentation', type=bool, default=True)
    parser.add_argument('--wandb_project', type=str, default="DL_assignment_2_eval")
    parser.add_argument('--wandb_entity', type=str, default="your_entity_name")
    return parser.parse_args()
# The main function
if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AdaptiveCNN(input_size=224, input_channels=3, **vars(args)).to(device)
    train_data, val_data = data_load("/kaggle/input/mydata/inaturalist_12K", args.data_augmentation)
    test_data = test_data_load("/kaggle/input/mydata/inaturalist_12K", args.data_augmentation)
    model_train_val_and_test(model, train_data, val_data, test_data, args)

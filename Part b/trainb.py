# importing the important libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import argparse
import wandb

# Set up
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Paths to files
TRAIN_DIR = '/kaggle/input/data-2/inaturalist_12K/train'
TEST_DIR = '/kaggle/input/data-2/inaturalist_12K/val'


IMAGE_SIZE = (224, 224)
NUM_CLASSES = 10


#  Function to set seed
def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)

# function to perfornm splitin training and validation
def stratified_split(dataset, val_fraction=0.2, seed=42):
    random.seed(seed)
    label_to_idx = defaultdict(list)
    for idx, (_, label) in enumerate(dataset.samples):
        label_to_idx[label].append(idx)

    train_indices, val_indices = [], []
    for label, indices in label_to_idx.items():
        n_val = int(len(indices) * val_fraction)
        random.shuffle(indices)
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    return Subset(dataset, train_indices), Subset(dataset, val_indices)

# function to prepare the data resising and normalising
def prepare_data(data_dir, val_fraction=0.2, use_augmentation=True):
    core_transforms = [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
    aug_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10)
    ]
    transform = transforms.Compose(aug_transforms + core_transforms) if use_augmentation else transforms.Compose(core_transforms)
    full_dataset = datasets.ImageFolder(data_dir, transform=transform)
    return stratified_split(full_dataset, val_fraction=val_fraction)


# function to load the test data
def load_test_data(test_dir):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return datasets.ImageFolder(test_dir, transform=transform)

# Function to load pretrained resnet 50
def initialize_model(num_classes=10, freeze_ratio=0.8):
    model = models.resnet50(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    params = list(model.parameters())
    freeze_limit = int(len(params) * freeze_ratio)
    for i, param in enumerate(params):
        param.requires_grad = (i >= freeze_limit)
    for param in model.fc.parameters():
        param.requires_grad = True

    return model.to(device)

# function to train the model and logging in wandb
def train_model(model, dataloaders, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.lr,
        weight_decay=config.l2_reg
    )

    wandb.watch(model, criterion, log="all", log_freq=100)

    for epoch in range(config.epochs):
        model.train()
        train_loss, total, correct = 0.0, 0, 0
        for xb, yb in dataloaders['train']:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            total += yb.size(0)
            correct += (output.argmax(1) == yb).sum().item()
        train_acc = correct / total
        avg_train_loss = train_loss / len(dataloaders['train'])

        model.eval()
        val_loss, total, correct = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in dataloaders['val']:
                xb, yb = xb.to(device), yb.to(device)
                output = model(xb)
                loss = criterion(output, yb)
                val_loss += loss.item()
                total += yb.size(0)
                correct += (output.argmax(1) == yb).sum().item()
        val_acc = correct / total
        avg_val_loss = val_loss / len(dataloaders['val'])

        wandb.log({
            'epoch': epoch + 1,
            'train_accuracy': train_acc * 100,
            'val_accuracy': val_acc * 100,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        })

        print(f"Epoch [{epoch+1}/{config.epochs}] | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

# function to evaluate on the test dataset
def evaluate_on_test(model, test_loader):
    model.eval()
    total, correct = 0, 0
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            output = model(xb)
            loss = criterion(output, yb)
            test_loss += loss.item()
            total += yb.size(0)
            correct += (output.argmax(1) == yb).sum().item()
    test_acc = correct / total
    avg_test_loss = test_loss / len(test_loader)
    wandb.log({'test_accuracy': test_acc * 100, 'test_loss': avg_test_loss})
    print(f"\n Test Accuracy: {test_acc:.4f} | Test Loss: {avg_test_loss:.4f}")


# The  MAIN function ( best hyperparametrs are set as dafault values)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ResNet50 on iNaturalist")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--freeze_ratio', type=float, default=0.8, help="Freeze ratio for pretrained model")
    parser.add_argument('--l2_reg', type=float, default=0.0005, help="L2 regularization (weight decay)")
    parser.add_argument('--data_aug', type=str, default='yes', choices=['yes', 'no'], help="Use data augmentation or not")
    parser.add_argument('--wandb_project', type=str, default='fine-tune-inaturalist', help="W&B project name")
    parser.add_argument('--wandb_entity', type=str, default=None, help="W&B entity (team or user)")
    args = parser.parse_args()

    wandb.login()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)
    set_seed()

    use_aug = args.data_aug == 'yes'
    train_set, val_set = prepare_data(TRAIN_DIR, val_fraction=0.2, use_augmentation=use_aug)
    test_set = load_test_data(TEST_DIR)

    dataloaders = {
        'train': DataLoader(train_set, batch_size=args.batch_size, shuffle=True),
        'val': DataLoader(val_set, batch_size=args.batch_size),
        'test': DataLoader(test_set, batch_size=args.batch_size)
    }

    model = initialize_model(num_classes=NUM_CLASSES, freeze_ratio=args.freeze_ratio)
    train_model(model, dataloaders, args)
    evaluate_on_test(model, dataloaders['test'])

    wandb.finish()

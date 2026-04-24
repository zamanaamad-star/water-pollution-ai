import os
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# Device (GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Settings
DATA_DIR = "water_pollution_dataset"
BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 1e-4
IMG_SIZE = 224


# Data transforms
def get_data_transforms():
    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return train_tfms, eval_tfms


# Load dataset
def load_data():
    train_tfms, eval_tfms = get_data_transforms()

    image_datasets = {
        "train": datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tfms),
        "val": datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=eval_tfms),
    }

    dataloaders = {
        "train": DataLoader(image_datasets["train"], batch_size=BATCH_SIZE, shuffle=True),
        "val": DataLoader(image_datasets["val"], batch_size=BATCH_SIZE, shuffle=False),
    }

    dataset_sizes = {
        "train": len(image_datasets["train"]),
        "val": len(image_datasets["val"]),
    }

    class_names = image_datasets["train"].classes

    return dataloaders, dataset_sizes, class_names


# Build model
def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model.to(DEVICE)


# Training loop
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 30)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double().item() / dataset_sizes[phase]

            print(f"{phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Save best model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\nBest Validation Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)

    return model


# Main function
def main():
    dataloaders, dataset_sizes, class_names = load_data()

    print("Classes:", class_names)

    model = build_model(num_classes=len(class_names))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    model = train_model(model, dataloaders, dataset_sizes, criterion, optimizer)

    # Save model
    Path("artifacts").mkdir(exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": class_names
    }, "artifacts/water_pollution_model.pth")

    print("\nModel saved to artifacts/water_pollution_model.pth")


if __name__ == "__main__":
    main()

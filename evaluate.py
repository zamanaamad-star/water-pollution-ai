import os
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "water_pollution_dataset"
MODEL_PATH = "artifacts/water_pollution_model.pth"
IMG_SIZE = 224
BATCH_SIZE = 16


def load_model(num_classes):
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, class_names


def main():
    tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    test_path = os.path.join(DATA_DIR, "test")
    if not os.path.exists(test_path):
        print("Test folder not found.")
        return

    try:
        test_ds = datasets.ImageFolder(test_path, transform=tfms)
    except FileNotFoundError as e:
        print("Test dataset not ready yet.")
        print(e)
        return

    if len(test_ds) == 0:
        print("No test images found yet.")
        return

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model, class_names = load_model(num_classes=len(test_ds.classes))

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.tight_layout()
    plt.savefig("artifacts/confusion_matrix.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()

import os
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "artifacts/water_pollution_model.pth"
IMG_SIZE = 224


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found yet. Train the model first.")
        return None, None

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


def predict_image(image_path):
    model, class_names = load_model()
    if model is None:
        return None, None

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None, None

    tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    x = tfms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()

    return class_names[pred_idx], float(probs[pred_idx])
def get_recommendation(label):
    recommendations = {
        "clean": (
            "No visible pollution detected.\n"
            "No action needed."
        ),

        "trash_debris": (
            "Possible solid waste contamination detected.\n"
            "Report here: https://mywaterquality.ca.gov/"
        ),

        "foam_or_sheen": (
            "Possible chemical contamination (foam or oily sheen).\n"
            "Resources:\n"
            "- https://www.waterboards.ca.gov/\n"
            "- https://www.waterboards.ca.gov/water_issues/programs/swamp/"
        ),

        "murky_or_discolored": (
            "Possible sediment or runoff pollution detected.\n"
            "Resources:\n"
            "- https://water.ca.gov/\n"
            "- https://mywaterquality.ca.gov/"
        )
    }

    return recommendations.get(label, "No recommendation available.")

if __name__ == "__main__":
    image_path = "example.jpg"
    label, conf = predict_image(image_path)

    if label is not None:
        print(f"\nPrediction: {label} ({conf:.3f})\n")
        print("Recommendation:")
        print(get_recommendation(label))

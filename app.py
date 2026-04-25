import os
import torch
import streamlit as st
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "artifacts/water_pollution_model.pth"
IMG_SIZE = 224


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


def get_recommendation(label):
    recommendations = {
        "clean": "No visible pollution detected. No action needed.",

        "trash_debris": (
            "Possible solid waste contamination detected.\n\n"
            "Report here:\nhttps://mywaterquality.ca.gov/"
        ),

        "foam_or_sheen": (
            "Possible chemical contamination (foam or oily sheen).\n\n"
            "Resources:\n"
            "https://www.waterboards.ca.gov/\n"
            "https://www.waterboards.ca.gov/water_issues/programs/swamp/"
        ),

        "murky_or_discolored": (
            "Possible sediment or runoff pollution detected.\n\n"
            "Resources:\n"
            "https://water.ca.gov/\n"
            "https://mywaterquality.ca.gov/"
        )
    }

    return recommendations.get(label, "No recommendation available.")


def predict_image(image, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    return class_names[pred_idx], float(probs[pred_idx])


# ---------------- STREAMLIT UI ----------------

st.set_page_config(page_title="Water Pollution AI", layout="centered")

st.title("🌊 Water Pollution AI")
st.write(
    "Upload a stream image to detect visible indicators of water pollution "
    "and get guidance on reporting resources."
)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

model, class_names = load_model()

if model is None:
    st.warning("Model not found yet. Train your model first using train.py.")
else:
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Analyze Image"):
            label, confidence = predict_image(image, model, class_names)

            st.subheader("Prediction")
            st.write(f"**Class:** {label}")
            st.write(f"**Confidence:** {confidence:.3f}")

            st.subheader("Recommendation")
            st.write(get_recommendation(label))

            st.caption(
                "This tool provides visual screening only and does not replace official water testing."
            )

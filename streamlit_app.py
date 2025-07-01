import streamlit as st
from PIL import Image
import torch
from digit_predictor import DigitCNN, predict_digit_pil, DEVICE

# Load model once
@st.cache_resource
def load_model():
    model = DigitCNN().to(DEVICE)
    model.load_state_dict(torch.load("mnist_cnn.pth", map_location=DEVICE))
    model.eval()
    return model

# UI layout
st.title("🧠 Digit Recognizer")
st.write("Upload a handwritten digit (28x28, grayscale preferred). The model will predict the number!")

uploaded_file = st.file_uploader("📤 Upload digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")  # Ensure grayscale
    st.image(image, caption="Uploaded Image", width=150)
    
    model = load_model()
    pred = predict_digit_pil(image, model)
    
    st.success(f"✅ Predicted Digit: **{pred}**")

"""
Digit Predictor – PyTorch implementation
======================================
A clean, modular training + inference pipeline for handwritten‐digit recognition
(MNIST‑like). Includes:
  • CNN with BatchNorm + Dropout
  • Training / evaluation loops with live metrics
  • Saved checkpoint (.pth)
  • Inference helper + optional Gradio UI that returns *only* the predicted digit
Author: ChatGPT (o3) – July 2025
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image, ImageOps

try:
    import gradio as gr  # Optional, only needed for the UI
except ImportError:
    gr = None

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────
DATA_DIR = Path("./data")
MODEL_DIR = Path("./checkpoints")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "mnist_cnn.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),       # Ensure single channel
    transforms.Resize((28, 28)),                        # MNIST size
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1.0 - x),               # Invert (white BG → black BG)
    transforms.Normalize((0.1307,), (0.3081,)),         # MNIST stats
])

# ────────────────────────────────────────────────────────────────────────────────
# Model Definition
# ────────────────────────────────────────────────────────────────────────────────
class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ────────────────────────────────────────────────────────────────────────────────
# Training & Evaluation
# ────────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    loss_total, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Eval", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss_total += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return loss_total / total, correct / total


# ────────────────────────────────────────────────────────────────────────────────
# Inference helper
# ────────────────────────────────────────────────────────────────────────────────
@torch.inference_mode()
def predict_digit_pil(img: Image.Image, model: nn.Module, transform=TRANSFORM) -> int:
    """Return predicted digit (0‑9) for a PIL image."""
    model.eval()
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    logits = model(img_tensor)
    pred = logits.argmax(dim=1).item()
    return pred


def predict_digit(img_input):
    """Wrapper for Gradio; accepts PIL or ndarray as provided by gr.Image."""
    if isinstance(img_input, str):
        img = Image.open(img_input)
    else:
        img = Image.fromarray(img_input) if not isinstance(img_input, Image.Image) else img_input
    return int(predict_digit_pil(img, MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))))


# ────────────────────────────────────────────────────────────────────────────────
# Main – CLI entry
# ────────────────────────────────────────────────────────────────────────────────
def main(args):
    # Data
    train_ds = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=TRANSFORM)
    test_ds = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model, loss, optimizer
    model = DigitCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch:02}/{args.epochs} – "
              f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.3%} | "
              f"val_loss: {test_loss:.4f}, val_acc: {test_acc:.3%}")
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✓ Saved new best model to {MODEL_PATH} (acc={best_acc:.3%})")

    print("Training complete. Best val accuracy:", f"{best_acc:.3%}")

    # Optionally launch Gradio
    if args.gradio and gr is not None:
        iface = gr.Interface(
            fn=lambda img: predict_digit_pil(img, model),
            inputs=gr.Image(source="upload", tool="sketch", type="pil", shape=(28, 28), label="Upload / Draw Digit"),
            outputs=gr.Number(label="Prediction"),
            title="MNIST Digit Predictor",
            description="Drop or draw a 0‑9 digit. The model returns the predicted class only."
        )
        iface.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN to classify handwritten digits (MNIST)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--gradio", action="store_true", help="Launch Gradio demo after training")
    args = parser.parse_args()
    main(args)
    
# ─── EXPORTS FOR STREAMLIT ──────────────────────────────────────────────
MODEL_PATH = "mnist_cnn.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


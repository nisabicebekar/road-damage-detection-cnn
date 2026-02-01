
import sys
import os
sys.path.append(os.path.abspath("."))

print("APP.PY ÇALIŞIYOR")

import torch
import gradio as gr
from PIL import Image
import torchvision.transforms as transforms

from src.models.cnn_model import SimpleCNN


DEVICE = "cpu"
MODEL_PATH = "outputs/model.pth"

model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model yüklendi (Gradio)")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def predict(image, threshold):
    """
    image: PIL Image
    threshold: float
    """
    image = image.convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

    if prob >= threshold:
        return (
            "🚧 DAMAGE DETECTED\n"
            f"Confidence: {prob:.2f}\n"
            f"Threshold: {threshold:.2f}"
        )
    else:
        return (
            "✅ NO DAMAGE\n"
            f"Confidence: {1 - prob:.2f}\n"
            f"Threshold: {threshold:.2f}"
        )


interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Yol Görüntüsü Yükle"),
        gr.Slider(
            minimum=0.3,
            maximum=0.9,
            value=0.75,
            step=0.05,
            label="Decision Threshold"
        )
    ],
    outputs=gr.Textbox(label="Tahmin Sonucu"),
    title="Yol Hasar Tespiti",
    description=(
        "Bu uygulama, yol görüntülerinde hasar (çatlak / bozulma) "
        "olup olmadığını tahmin eder.\n\n"
        
    )
)


if __name__ == "__main__":
    interface.launch()
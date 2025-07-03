from ultralytics import YOLO
import numpy as np
import tempfile
from PIL import Image
import torch

# Gebruik CPU als er geen GPU is
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Laad het modelbestand (zorg dat dit klopt met bestandsnaam!)
model = YOLO("../Modellen/getrainde modellen/yolov8n_nest_50epochs.pt")
import io
from PIL import ImageDraw

def run_model(pil_image: Image.Image):
    # Converteer indien nodig
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")

    # Tijdelijk opslaan
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        pil_image.save(temp.name)
        image_path = temp.name

    # YOLOv8 voorspelling
    results = model.predict(source=image_path, device=device, save=False)

    # Teken de bounding boxes op de afbeelding
    drawn_img = pil_image.copy()
    draw = ImageDraw.Draw(drawn_img)

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]  # coördinaten
        label = int(box.cls[0])
        conf = float(box.conf[0])
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{label} ({conf:.2f})", fill="red")

    # Resultaten teruggeven
    return {
        "n_boxes": len(results[0].boxes),
        "labels": [int(c) for c in results[0].boxes.cls.tolist()],
        "confidences": [float(c) for c in results[0].boxes.conf.tolist()],
        "image_with_boxes": drawn_img
    }

from ultralytics import YOLO
from PIL import Image, ImageDraw
import tempfile
import torch

# Selecteer device (CUDA of CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Laad het YOLOv8-model
model = YOLO("Modellen/getrainde modellen/yolov8n_nest_50epochs.pt")


def convert_image(image: Image.Image) -> Image.Image:
    """
        Converteer een RGBA-afbeelding naar RGB indien nodig.

        :param image: PIL Image object (RGBA of RGB)
        :return: PIL Image object in RGB
    """
    return image.convert("RGB") if image.mode == "RGBA" else image


def save_temp_image(image: Image.Image) -> str:
    """
        Sla de afbeelding tijdelijk op als JPEG-bestand.

        :param image: PIL Image object
        :return: Pad naar het tijdelijk opgeslagen bestand
    """
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        image.save(temp.name)
        return temp.name


def predict_with_model(image_path: str):
    """
        Voer YOLOv8-predictie uit op een afbeelding via het opgegeven pad.

        :param image_path: Pad naar afbeelding
        :return: YOLO predictieobject (eerste resultaat)
    """
    return model.predict(source=image_path, device=device, save=False)[0]


def draw_boxes_on_image(image: Image.Image, results) -> Image.Image:
    """
        Teken bounding boxes met labels en confidence scores op een afbeelding.

        :param image: Originele afbeelding (PIL Image)
        :param results: YOLOv8 result object met .boxes
        :return: Annotated afbeelding (PIL Image)
    """
    drawn_img = image.copy()
    draw = ImageDraw.Draw(drawn_img)

    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        label = int(box.cls[0])
        conf = float(box.conf[0])
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1), f"{label} ({conf:.2f})", fill="red")

    return drawn_img


def run_model(pil_image: Image.Image) -> dict:
    """
        Hoofdfunctie voor het verwerken van een afbeelding:
        converteert deze, slaat tijdelijk op, voert predictie uit,
        en tekent resultaten.

        :param pil_image: PIL Image object
        :return: Dict met aantal boxes, labels, confidence scores en gemarkeerde afbeelding
    """
    image_rgb = convert_image(pil_image)
    temp_path = save_temp_image(image_rgb)
    results = predict_with_model(temp_path)
    annotated_image = draw_boxes_on_image(image_rgb, results)

    return {
        "n_boxes": len(results.boxes),
        "labels": [int(c) for c in results.boxes.cls.tolist()],
        "confidences": [float(c) for c in results.boxes.conf.tolist()],
        "image_with_boxes": annotated_image
    }

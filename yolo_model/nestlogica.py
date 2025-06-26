
import random
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_erosion
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

PIXEL_SCALE = 0.0175  # meter per pixel

rules = {
    'huismus':     {'min_h': 3, 'max_h': 10},
    'gierzwaluw':  {'min_h': 6, 'max_h': 40},
    'vleermuis':   {'min_h': 3, 'max_h': 50}
}

def analyseer_nestlocaties(pil_image: Image.Image, species: str, model_path: str, icon_paths: dict):
    if species not in rules:
        raise ValueError(f"Ongeldig dier: {species}")

    model = YOLO(model_path)
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
    img_width, img_height = pil_image.size
    results = model.predict(pil_image, save=False)[0]

    img = pil_image
    facade_mask = np.zeros((img_height, img_width), dtype=bool)
    window_mask = np.zeros((img_height, img_width), dtype=bool)

    for box in results.boxes:
        cls = int(box.cls.item())
        label = results.names[cls].lower()
        if label == 'window':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            window_mask[y1:y2+1, x1:x2+1] = True

    if results.masks:
        for seg, cls in zip(results.masks.data, results.boxes.cls):
            if results.names[int(cls.item())].lower() == 'facade':
                mask = seg.cpu().numpy() > 0.5
                resized = np.array(Image.fromarray(mask.astype(np.uint8)*255).resize((img_width, img_height), resample=Image.NEAREST))
                facade_mask |= resized.astype(bool)

    # 60 cm marge
    pixels_margin = int(np.ceil(0.60 / PIXEL_SCALE))
    facade_inner = binary_erosion(facade_mask, structure=np.ones((3, 3)), iterations=pixels_margin)

    # 1 meter afstand tot ramen/deuren
    distance_from_windows = distance_transform_edt(~window_mask) * PIXEL_SCALE
    safe_from_windows = distance_from_windows >= 1.0

    # Hoogte-filter
    min_h = rules[species]['min_h']
    max_h = rules[species]['max_h']
    y_coords = np.arange(img_height).reshape(-1, 1)
    height_from_bottom = (img_height - y_coords) * PIXEL_SCALE
    height_mask = (height_from_bottom >= min_h) & (height_from_bottom <= max_h)
    height_mask = np.repeat(height_mask, img_width, axis=1)

    allowed_mask = facade_inner & safe_from_windows & height_mask

    min_distance_m = 3.0
    min_distance_px = int(np.ceil(min_distance_m / PIXEL_SCALE))
    all_coords = np.argwhere(allowed_mask)
    selected_nests = []

    for y, x in all_coords:
        if len(selected_nests) == 5:
            break
        if all(np.hypot(x - sx, y - sy) >= min_distance_px for sy, sx in selected_nests):
            selected_nests.append((y, x))

    output_img = np.array(img.convert("RGB"))
    overlay = output_img.copy()
    overlay[allowed_mask] = [0, 255, 0]
    blended = (0.6 * output_img + 0.4 * overlay).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(blended)

    if species in icon_paths:
        icon_img = plt.imread(icon_paths[species])
        for y, x in selected_nests:
            imagebox = OffsetImage(icon_img, zoom=0.05)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)

    ax.set_title(f"Nestlocaties voor: {species} (≥3m afstand)")
    ax.axis('off')

    locaties = []
    for y, x in selected_nests:
        hoogte = y * PIXEL_SCALE
        locaties.append({'x': x, 'y': y, 'hoogte_m': round(hoogte, 2)})

    return fig, locaties

import random
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_erosion
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def load_model(model_path):
    return YOLO(model_path)

def load_image(image_path):
    return Image.open(image_path)

def detect_objects(model, image_path):
    results = model.predict(source=image_path, save=False)[0]
    return results

def create_masks(results, image_shape):
    img_height, img_width = image_shape
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
                resized = np.array(Image.fromarray(mask.astype(np.uint8)*255).resize(
                    (img_width, img_height), resample=Image.NEAREST))
                facade_mask |= resized.astype(bool)

    return facade_mask, window_mask

def create_allowed_mask(facade_mask, window_mask, img_shape, pixel_scale, min_h, max_h):
    img_height, img_width = img_shape
    pixels_margin = int(np.ceil(0.60 / pixel_scale))
    facade_inner = binary_erosion(facade_mask, structure=np.ones((3, 3)), iterations=pixels_margin)

    distance_from_windows = distance_transform_edt(~window_mask) * pixel_scale
    safe_from_windows = distance_from_windows >= 1.0

    y_coords = np.arange(img_height).reshape(-1, 1)
    height_from_bottom = (img_height - y_coords) * pixel_scale
    height_mask = (height_from_bottom >= min_h) & (height_from_bottom <= max_h)
    height_mask = np.repeat(height_mask, img_width, axis=1)

    return facade_inner & safe_from_windows & height_mask

def select_nest_locations(mask, pixel_scale, min_distance_m=3.0, max_nests=5):
    min_distance_px = int(np.ceil(min_distance_m / pixel_scale))
    all_coords = np.argwhere(mask)
    selected_nests = []

    for y, x in all_coords:
        if len(selected_nests) == max_nests:
            break
        if all(np.hypot(x - sx, y - sy) >= min_distance_px for sy, sx in selected_nests):
            selected_nests.append((y, x))

    return selected_nests

def visualize_nests(image, allowed_mask, selected_nests, icon_path, species, pixel_scale):
    output_img = np.array(image.convert("RGB"))
    overlay = output_img.copy()
    overlay[allowed_mask] = [0, 255, 0]

    blended = (0.6 * output_img + 0.4 * overlay).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(blended)

    icon_img = plt.imread(icon_path)
    for y, x in selected_nests:
        imagebox = OffsetImage(icon_img, zoom=0.05)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        ax.add_artist(ab)

    ax.set_title(f"Nestlocaties voor: {species} (≥3m afstand)")
    ax.axis('off')
    plt.show()

def print_results(selected_nests, species, pixel_scale):
    print(f"\nNestlocaties voor {species} (min. 3m tussenafstand):")
    for i, (y, x) in enumerate(selected_nests, 1):
        hoogte = y * pixel_scale
        print(f"Nest {i}: x={x}, y={y} → hoogte ≈ {hoogte:.2f} m")
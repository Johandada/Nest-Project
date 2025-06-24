import random
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_erosion
from matplotlib.offsetbox import OffsetImage, AnnotationBbox  # ← toegevoegd voor iconen

# 📂 Pad instellingen
IMAGE_PATH = r"C:\Users\moham\Nest-Project\nest_data\test\images\test_3.jpg"
MODEL_PATH = r"C:\Users\moham\Nest-Project\yolov8n_nest_50epochs.pt"

# 📐 Schaal
PIXEL_SCALE = 0.0175  # meter per pixel

# 🐦 Nestregels
rules = {
    'huismus':     {'min_h': 3, 'max_h': 10},
    'gierzwaluw':  {'min_h': 6, 'max_h': 40},
    'vleermuis':   {'min_h': 3, 'max_h': 50}
}

# 👤 Gebruiker kiest dier
species = input("Kies een dier (huismus, gierzwaluw, vleermuis): ").strip().lower()
if species not in rules:
    raise ValueError("Ongeldig dier gekozen.")

# 🔍 YOLO-model laden en detecteren
model = YOLO(MODEL_PATH)
results = model.predict(source=IMAGE_PATH, save=False)[0]

# 📷 Afbeelding laden
img = Image.open(IMAGE_PATH)
img_width, img_height = img.size

# Maskers initialiseren
facade_mask = np.zeros((img_height, img_width), dtype=bool)
window_mask = np.zeros((img_height, img_width), dtype=bool)

# Detectie verwerken
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

# ➖ 60 cm marge op gevel-rand
pixels_margin = int(np.ceil(0.60 / PIXEL_SCALE))
facade_inner = binary_erosion(facade_mask, structure=np.ones((3, 3)), iterations=pixels_margin)

# 🚪 1 meter afstand tot ramen/deuren houden
distance_from_windows = distance_transform_edt(~window_mask) * PIXEL_SCALE
safe_from_windows = distance_from_windows >= 1.0

# 📏 Hoogte-filter
min_h = rules[species]['min_h']
max_h = rules[species]['max_h']

y_coords = np.arange(img_height).reshape(-1, 1)  # van boven (0) naar onder (img_height-1)
height_from_bottom = (img_height - y_coords) * PIXEL_SCALE  # onderaan = 0 m, boven = hoger

height_mask = (height_from_bottom >= min_h) & (height_from_bottom <= max_h)
height_mask = np.repeat(height_mask, img_width, axis=1)

# ✅ Combineer alles
allowed_mask = facade_inner & safe_from_windows & height_mask

# 🐣 Selecteer 5 nesten met minimaal 3 meter afstand tussen elkaar
min_distance_m = 3.0
min_distance_px = int(np.ceil(min_distance_m / PIXEL_SCALE))
all_coords = np.argwhere(allowed_mask)
selected_nests = []

for y, x in all_coords:
    if len(selected_nests) == 5:
        break
    if all(np.hypot(x - sx, y - sy) >= min_distance_px for sy, sx in selected_nests):
        selected_nests.append((y, x))

# 🎨 Visualisatie
output_img = np.array(img.convert("RGB"))
overlay = output_img.copy()
overlay[allowed_mask] = [0, 255, 0]  # Groen waar toegestaan

blended = (0.6 * output_img + 0.4 * overlay).astype(np.uint8)

fig, ax = plt.subplots(figsize=(12, 8))
ax.imshow(blended)

# 🖼️ Nesticonen toevoegen
icon_paths = {
    'huismus':     r"C:\Users\moham\Nest-Project\huismus.png",
    'gierzwaluw':  r"C:\Users\moham\Nest-Project\gierzwaluw.png",
    'vleermuis':   r"C:\Users\moham\Nest-Project\vleermuis.png"
}

icon_img = plt.imread(icon_paths[species])
for y, x in selected_nests:
    imagebox = OffsetImage(icon_img, zoom=0.05)  # zoom kan je aanpassen indien nodig
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)

ax.set_title(f"Nestlocaties voor: {species} (≥3m afstand)")
ax.axis('off')
plt.show()

# 📄 Resultaten printen
print(f"\nNestlocaties voor {species} (min. 3m tussenafstand):")
for i, (y, x) in enumerate(selected_nests, 1):
    hoogte = y * PIXEL_SCALE
    print(f"Nest {i}: x={x}, y={y} → hoogte ≈ {hoogte:.2f} m")

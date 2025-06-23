import random
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# 📂 Paden
IMAGE_PATH = r"C:\Users\moham\Nest-Project\nest_data\test\images\rijwoning-6-_png.rf.a677db79b71b58093105ad296ef13e93.jpg"
MODEL_PATH = r"C:\Users\moham\Nest-Project\yolov8n_nest_50epochs.pt"

# 📐 Schaal
PIXEL_SCALE = 0.0175  # meter per pixel

# 🐦 Nestregels
rules = {
    'huismus':     {'min_h': 3, 'max_h': 10, 'aantal': 5},
    'gierzwaluw':  {'min_h': 6, 'max_h': 40, 'aantal': 5},
    'vleermuis':   {'min_h': 3, 'max_h': 50, 'afstand_tot_raam': 1, 'aantal': 3}
}

# 🔍 Model laden
model = YOLO(MODEL_PATH)
results = model.predict(source=IMAGE_PATH, save=False)[0]

# 📷 Afbeelding
img = Image.open(IMAGE_PATH)
img_width, img_height = img.size

# 🧱 Masks & ramen verzamelen
facade_mask = np.zeros((img_height, img_width), dtype=bool)
window_boxes = []

for i, box in enumerate(results.boxes):
    cls = int(box.cls.item())
    label = results.names[cls].lower()

    if label == 'window':
        x1, y1, x2, y2 = box.xyxy[0]
        window_boxes.append((x1.item(), y1.item(), x2.item(), y2.item()))

# ➕ Combineer alle façade-masks
if results.masks:
    for seg, cls in zip(results.masks.data, results.boxes.cls):
        label = results.names[int(cls.item())].lower()
        if label == 'facade':
            mask = seg.cpu().numpy()
            mask = np.array(mask > 0.5)
            # Schaal mask naar originele resolutie
            mask_resized = np.array(Image.fromarray(mask).resize((img_width, img_height)))
            facade_mask |= mask_resized

# 🐣 Nestplaatsen
nesten = {'huismus': [], 'gierzwaluw': [], 'vleermuis': []}
max_pogingen = 5000

for soort, regel in rules.items():
    nodig = regel['aantal']
    poging = 0

    while len(nesten[soort]) < nodig and poging < max_pogingen:
        poging += 1
        x = random.randint(0, img_width - 1)
        y = random.randint(0, img_height - 1)
        if not facade_mask[y, x]:
            continue

        hoogte_m = y * PIXEL_SCALE
        if not (regel['min_h'] <= hoogte_m <= regel['max_h']):
            continue

        if soort == 'vleermuis':
            dichtbij_raam = False
            for rx1, ry1, rx2, ry2 in window_boxes:
                midden_raam = ((rx1 + rx2) / 2, (ry1 + ry2) / 2)
                dx = abs(x - midden_raam[0]) * PIXEL_SCALE
                dy = abs(y - midden_raam[1]) * PIXEL_SCALE
                afstand = (dx**2 + dy**2)**0.5
                if afstand < regel['afstand_tot_raam']:
                    dichtbij_raam = True
                    break
            if dichtbij_raam:
                continue

        nesten[soort].append((x, y))

# 🎨 Visualisatie
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.title("Nestlocaties binnen façade (segmentatie)")

# 🖼️ Overlay icons in plaats van scatter
icon_paths = {
    'huismus':    r"C:\Users\moham\Nest-Project\huismus.png",
    'gierzwaluw': r"C:\Users\moham\Nest-Project\gierzwaluw.png",
    'vleermuis':  r"C:\Users\moham\Nest-Project\vleermuis.png"
}

for soort, punten in nesten.items():
    icon = plt.imread(icon_paths[soort])
    for x, y in punten:
        imagebox = OffsetImage(icon, zoom=0.05)  # pas zoom aan als nodig
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        plt.gca().add_artist(ab)

plt.show()

# 🖨️ Print resultaten
for soort, punten in nesten.items():
    print(f"\n🔹 {soort.capitalize()} ({len(punten)} locaties):")
    for i, (x, y) in enumerate(punten):
        hoogte = y * PIXEL_SCALE
        print(f"  - Punt {i+1}: x={x}, y={y} → hoogte ≈ {hoogte:.2f} m")
import random
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# 📂 Paden
IMAGE_PATH = r"C:\Users\moham\Nest-Project\nest_data\test\images\rijwoning-6-_png.rf.a677db79b71b58093105ad296ef13e93.jpg"
MODEL_PATH = r"C:\Users\moham\Nest-Project\yolov8n_nest_50epochs.pt"

# 📐 Schaal
PIXEL_SCALE = 0.0175  # meter per pixel

# 🐦 Nestregels
rules = {
    'huismus':     {'min_h': 3, 'max_h': 10, 'aantal': 5},
    'gierzwaluw':  {'min_h': 6, 'max_h': 40, 'aantal': 5},
    'vleermuis':   {'min_h': 3, 'max_h': 50, 'afstand_tot_raam': 1, 'aantal': 3}
}

# 🔍 Model laden
model = YOLO(MODEL_PATH)
results = model.predict(source=IMAGE_PATH, save=False)[0]

# 📷 Afbeelding
img = Image.open(IMAGE_PATH)
img_width, img_height = img.size

# 🧱 Masks & ramen verzamelen
facade_mask = np.zeros((img_height, img_width), dtype=bool)
window_boxes = []

for i, box in enumerate(results.boxes):
    cls = int(box.cls.item())
    label = results.names[cls].lower()

    if label == 'window':
        x1, y1, x2, y2 = box.xyxy[0]
        window_boxes.append((x1.item(), y1.item(), x2.item(), y2.item()))

# ➕ Combineer alle façade-masks
if results.masks:
    for seg, cls in zip(results.masks.data, results.boxes.cls):
        label = results.names[int(cls.item())].lower()
        if label == 'facade':
            mask = seg.cpu().numpy()
            mask = np.array(mask > 0.5)
            # Schaal mask naar originele resolutie
            mask_resized = np.array(Image.fromarray(mask).resize((img_width, img_height)))
            facade_mask |= mask_resized

# 🐣 Nestplaatsen
nesten = {'huismus': [], 'gierzwaluw': [], 'vleermuis': []}
max_pogingen = 5000

for soort, regel in rules.items():
    nodig = regel['aantal']
    poging = 0

    while len(nesten[soort]) < nodig and poging < max_pogingen:
        poging += 1
        x = random.randint(0, img_width - 1)
        y = random.randint(0, img_height - 1)
        if not facade_mask[y, x]:
            continue

        hoogte_m = y * PIXEL_SCALE
        if not (regel['min_h'] <= hoogte_m <= regel['max_h']):
            continue

        if soort == 'vleermuis':
            dichtbij_raam = False
            for rx1, ry1, rx2, ry2 in window_boxes:
                midden_raam = ((rx1 + rx2) / 2, (ry1 + ry2) / 2)
                dx = abs(x - midden_raam[0]) * PIXEL_SCALE
                dy = abs(y - midden_raam[1]) * PIXEL_SCALE
                afstand = (dx**2 + dy**2)**0.5
                if afstand < regel['afstand_tot_raam']:
                    dichtbij_raam = True
                    break
            if dichtbij_raam:
                continue

        nesten[soort].append((x, y))

# 🎨 Visualisatie
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.title("Nestlocaties en façade-masker")

# 🔴 Overlay façade-mask (semi-transparant rood)
mask_overlay = np.zeros((img_height, img_width, 4), dtype=np.uint8)  # RGBA
mask_overlay[facade_mask] = [255, 0, 0, 100]  # Rood met alpha = 100/255
plt.imshow(mask_overlay)

# 🖼️ Overlay icons
icon_paths = {
    'huismus':    r"C:\Users\moham\Nest-Project\huismus.png",
    'gierzwaluw': r"C:\Users\moham\Nest-Project\gierzwaluw.png",
    'vleermuis':  r"C:\Users\moham\Nest-Project\vleermuis.png"
}

for soort, punten in nesten.items():
    icon = plt.imread(icon_paths[soort])
    for x, y in punten:
        imagebox = OffsetImage(icon, zoom=0.05)
        ab = AnnotationBbox(imagebox, (x, y), frameon=False)
        plt.gca().add_artist(ab)

plt.show()

# 🖨️ Print resultaten
for soort, punten in nesten.items():
    print(f"\n🔹 {soort.capitalize()} ({len(punten)} locaties):")
    for i, (x, y) in enumerate(punten):
        hoogte = y * PIXEL_SCALE
        print(f"  - Punt {i+1}: x={x}, y={y} → hoogte ≈ {hoogte:.2f} m")

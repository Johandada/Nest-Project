
import random
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt

PIXEL_SCALE = 0.0175  # meter per pixel

rules = {
    'huismus':     {'min_h': 3, 'max_h': 10, 'aantal': 5},
    'gierzwaluw':  {'min_h': 6, 'max_h': 40, 'aantal': 5},
    'vleermuis':   {'min_h': 3, 'max_h': 50, 'afstand_tot_raam': 1, 'aantal': 3}
}

def analyseer(pil_image: Image.Image, model_path: str):
    model = YOLO(model_path)
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
    img_width, img_height = pil_image.size
    results = model.predict(pil_image, save=False)[0]

    facade_mask = np.zeros((img_height, img_width), dtype=bool)
    window_boxes = []

    for box in results.boxes:
        cls = int(box.cls.item())
        label = results.names[cls].lower()
        if label == 'window':
            x1, y1, x2, y2 = box.xyxy[0]
            window_boxes.append((x1.item(), y1.item(), x2.item(), y2.item()))

    if results.masks:
        for seg, cls in zip(results.masks.data, results.boxes.cls):
            label = results.names[int(cls.item())].lower()
            if label == 'facade':
                mask = seg.cpu().numpy()
                mask = np.array(mask > 0.5)
                mask_resized = np.array(Image.fromarray(mask).resize((img_width, img_height)))
                facade_mask |= mask_resized

    nesten = {'huismus': [], 'gierzwaluw': [], 'vleermuis': []}
    for soort, regel in rules.items():
        nodig = regel['aantal']
        poging = 0
        while len(nesten[soort]) < nodig and poging < 5000:
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
                    mx = (rx1 + rx2) / 2
                    my = (ry1 + ry2) / 2
                    dx = abs(x - mx) * PIXEL_SCALE
                    dy = abs(y - my) * PIXEL_SCALE
                    afstand = (dx**2 + dy**2)**0.5
                    if afstand < regel['afstand_tot_raam']:
                        dichtbij_raam = True
                        break
                if dichtbij_raam:
                    continue
            nesten[soort].append((x, y))

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(pil_image)
    plt.axis('off')
    plt.title("Nestlocaties binnen façade (segmentatie)")
    kleuren = {'huismus': 'blue', 'gierzwaluw': 'green', 'vleermuis': 'red'}
    symbolen = {'huismus': 'o', 'gierzwaluw': '^', 'vleermuis': 's'}
    for soort, punten in nesten.items():
        for x, y in punten:
            plt.scatter(x, y, c=kleuren[soort], marker=symbolen[soort], label=soort, s=60, edgecolors='black')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()

    # Print
    for soort, punten in nesten.items():
        print(f"\n🔹 {soort.capitalize()} ({len(punten)} locaties):")
        for i, (x, y) in enumerate(punten):
            hoogte = y * PIXEL_SCALE
            print(f"  - Punt {i+1}: x={x}, y={y} → hoogte ≈ {hoogte:.2f} m")

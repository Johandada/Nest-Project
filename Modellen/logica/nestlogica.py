import numpy as np
from PIL import Image
from ultralytics import YOLO
from scipy.ndimage import distance_transform_edt, binary_erosion
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

PIXEL_SCALE = 0.0175  # meter per pixel

# Hoogteregels per soort
rules = {
    'huismus': {'min_h': 3, 'max_h': 10},
    'gierzwaluw': {'min_h': 6, 'max_h': 40},
    'vleermuis': {'min_h': 3, 'max_h': 50}
}


def voorspel_met_model(pil_image, model_path):
    """
        Voert een YOLOv8-predictie uit op een PIL-afbeelding.

        :param pil_image: PIL Image van de gevel
        :param model_path: Pad naar YOLOv8-model (.pt-bestand)
        :return: YOLOv8 predictie-output (results[0])
    """
    model = YOLO(model_path)
    if pil_image.mode == "RGBA":
        pil_image = pil_image.convert("RGB")
    results = model.predict(pil_image, save=False)[0]
    return results


def genereer_gevel_en_venster_maskers(results, img_shape):
    """
        Genereert maskers voor gevel (facade) en ramen/deuren (windows).

        :param results: YOLOv8 result-object
        :param img_shape: Tuple (hoogte, breedte)
        :return: Tuple (facade_mask, window_mask) als booleaanse arrays
    """
    h, w = img_shape
    facade_mask = np.zeros((h, w), dtype=bool)
    window_mask = np.zeros((h, w), dtype=bool)

    # Ramen detecteren via bounding boxes
    for box in results.boxes:
        cls = int(box.cls.item())
        label = results.names[cls].lower()
        if label == 'window':
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            window_mask[y1:y2 + 1, x1:x2 + 1] = True

    # Gevel detecteren via maskers
    if results.masks:
        for seg, cls in zip(results.masks.data, results.boxes.cls):
            if results.names[int(cls.item())].lower() == 'facade':
                mask = seg.cpu().numpy() > 0.5
                resized = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((w, h), resample=Image.NEAREST))
                facade_mask |= resized.astype(bool)

    return facade_mask, window_mask


def filter_geschikte_pixelgebieden(facade_mask, window_mask, img_shape, species):
    """
        Berekent geschikte pixels voor nestkasten op basis van gevel, hoogte en afstand.

        :param facade_mask: Gevelmasker (bool array)
        :param window_mask: Ramenmasker (bool array)
        :param img_shape: Tuple (hoogte, breedte)
        :param species: Naam van de diersoort
        :return: Booleaanse mask van toegestane pixels
    """
    h, w = img_shape

    # 60 cm marge vanaf gevelrand
    pixels_margin = int(np.ceil(0.60 / PIXEL_SCALE))
    facade_inner = binary_erosion(facade_mask, structure=np.ones((3, 3)), iterations=pixels_margin)

    # Bereken afstand tot ramen (in meters)
    distance_from_windows = distance_transform_edt(~window_mask) * PIXEL_SCALE
    safe_from_windows = distance_from_windows >= 1.0

    # Hoogtebereik
    min_h = rules[species]['min_h']
    max_h = rules[species]['max_h']
    y_coords = np.arange(h).reshape(-1, 1)
    height_from_bottom = (h - y_coords) * PIXEL_SCALE
    height_mask = (height_from_bottom >= min_h) & (height_from_bottom <= max_h)
    height_mask = np.repeat(height_mask, w, axis=1)

    # Alleen waar alles voldoet
    allowed_mask = facade_inner & safe_from_windows & height_mask
    return allowed_mask


def selecteer_nestlocaties(allowed_mask, max_nesten=5, min_distance_m=3.0):
    """
        Selecteert maximaal `max_nesten` locaties die minimaal `min_distance_m` uit elkaar liggen.

        :param allowed_mask: Mask van geldige nestposities (bool)
        :param max_nesten: Aantal nesten dat je wilt plaatsen
        :param min_distance_m: Minimale afstand tussen nesten in meters
        :return: Lijst van (y, x)-coördinaten van nesten
    """
    min_distance_px = int(np.ceil(min_distance_m / PIXEL_SCALE))
    all_coords = np.argwhere(allowed_mask)
    selected_nests = []

    for y, x in all_coords:
        if len(selected_nests) == max_nesten:
            break
        if all(np.hypot(x - sx, y - sy) >= min_distance_px for sy, sx in selected_nests):
            selected_nests.append((y, x))

    return selected_nests


def visualiseer_nestlocaties(img, allowed_mask, selected_nests, species, icon_paths):
    """
        Visualiseert de nestlocaties op de afbeelding met groene overlay en icoontjes.

        :param img: PIL Image object
        :param allowed_mask: Masker van toegestane gebieden
        :param selected_nests: Lijst van nestlocaties (y, x)
        :param species: Diersoort (voor icon)
        :param icon_paths: Dict met paden naar icons per soort
        :return: Tuple (figuur, lijst van locaties met x/y/hoogte)
    """
    output_img = np.array(img.convert("RGB"))
    overlay = output_img.copy()
    overlay[allowed_mask] = [0, 255, 0]  # markeer toegestaan gebied
    blended = (0.6 * output_img + 0.4 * overlay).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(blended)

    # Voeg icoontjes toe per nest
    if species in icon_paths:
        icon_img = plt.imread(icon_paths[species])
        for y, x in selected_nests:
            imagebox = OffsetImage(icon_img, zoom=0.05)
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)

    ax.set_title(f"Nestlocaties voor: {species} (\u22653m afstand)")
    ax.axis('off')

    locaties = [{'x': x, 'y': y, 'hoogte_m': round(y * PIXEL_SCALE, 2)} for y, x in selected_nests]
    return fig, locaties


def analyseer_nestlocaties(pil_image, species, model_path, icon_paths):
    """
        Voert de volledige nestkastanalyse uit op basis van een afbeelding en soort.

        :param pil_image: Geüploade gevelafbeelding (PIL)
        :param species: Naam van de diersoort (huismus, gierzwaluw, vleermuis)
        :param model_path: Pad naar YOLO-model
        :param icon_paths: Dict met paths naar diersoorticons
        :return: Tuple (figuur met visualisatie, lijst met locaties)
    """
    if species not in rules:
        raise ValueError(f"Ongeldig dier: {species}")

    results = voorspel_met_model(pil_image, model_path)
    img_width, img_height = pil_image.size
    facade_mask, window_mask = genereer_gevel_en_venster_maskers(results, (img_height, img_width))
    allowed_mask = filter_geschikte_pixelgebieden(facade_mask, window_mask, (img_height, img_width), species)
    geselecteerde = selecteer_nestlocaties(allowed_mask)
    figuur, locaties = visualiseer_nestlocaties(pil_image, allowed_mask, geselecteerde, species, icon_paths)
    return figuur, locaties

import cv2
import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator


def laad_sam_model(checkpoint_path="models/sam_vit_h_4b8939.pth", model_type="vit_h"):
    """
    Laad het Segment Anything Model (SAM) en initialiseer een SamPredictor.

    Parameters:
    - checkpoint_path (str): Pad naar het modelbestand (.pth).
    - model_type (str): Type ViT-model (bijv. 'vit_h').

    Returns:
    - sam: Het geladen SAM-model.
    - predictor: Een SamPredictor-object om segmentatie uit te voeren.
    """
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    predictor = SamPredictor(sam)
    return sam, predictor


def laad_afbeelding_en_prepareer(image_path, predictor):
    """
    Laad een afbeelding en converteer deze naar RGB. Bereid de afbeelding voor gebruik met SAM.

    Parameters:
    - image_path (str): Pad naar de invoerafbeelding.
    - predictor: SamPredictor-object.

    Returns:
    - image: Originele afbeelding in BGR.
    - image_rgb: Afbeelding in RGB-formaat geschikt voor SAM.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    return image, image_rgb


def genereer_punten(h, w, mode):
    """
    Genereer klikpunten voor het SAM-model, afhankelijk van het doel (dak of gebouw).

    Parameters:
    - h (int): Hoogte van de afbeelding.
    - w (int): Breedte van de afbeelding.
    - mode (str): 'dak' of 'gebouw'.

    Returns:
    - punten (np.ndarray): Array met x, y-coördinaten.
    - labels (np.ndarray): Array met labels (1 = positief punt).
    """
    center_x, center_y = w // 2, h // 2
    if mode == "dak":
        return np.array([[center_x, center_y - 250],
                         [center_x - 100, center_y - 250],
                         [center_x + 100, center_y - 250],
                         [center_x, center_y - 200],
                         [center_x - 100, center_y - 200],
                         [center_x + 100, center_y - 200],
                         [center_x, center_y - 150],
                         [center_x - 100, center_y - 150],
                         [center_x + 100, center_y - 150]]), np.ones(9)

    elif mode == "gebouw":
        offset_x, offset_y = 60, 120
        punten = np.array([
            [center_x + dx, center_y + dy]
            for dx in [-offset_x, 0, offset_x]
            for dy in [-offset_y, 0, offset_y]
        ])

        return punten, np.ones(len(punten))


def predict_mask(predictor, punten, labels):
    """
    Voorspel het beste segmentatiemasker op basis van gegeven klikpunten.

    Parameters:
    - predictor (SamPredictor): De SAM-predictor die segmentatie uitvoert.
    - punten (np.ndarray): Coördinaten van klikpunten (shape: [n, 2]).
    - labels (np.ndarray): Labels voor de klikpunten (meestal allemaal 1 voor positieve prompts).

    Returns:
    - beste_mask (np.ndarray): Het masker (boolean array) met de hoogste segmentatiescore.
    """
    masks, scores, _ = predictor.predict(
        point_coords=punten,
        point_labels=labels,
        multimask_output=True
    )

    beste_mask = masks[np.argmax(scores)]
    return beste_mask


def toon_masker(mask, punten, titel="Masker"):
    """
    Visualiseer een segmentatiemasker met de bijbehorende klikpunten.

    Parameters:
    - mask (np.ndarray): Een binair masker (2D-array) dat het segment voorstelt.
    - punten (np.ndarray): De klikpunten (meestal van gebruiker of automatisch gegenereerd).
    - titel (str): Titel die boven de plot wordt weergegeven.

    Returns:
    - None: De functie toont een plot met matplotlib, zonder iets te retourneren.
    """
    plt.imshow(mask)
    plt.scatter(punten[:, 0], punten[:, 1], color="blue", s=40, marker="o")
    plt.title(titel)
    plt.axis("off")
    plt.show()


def maak_rgba_gebouw(image_rgb, mask):
    """
    Genereer een afbeelding met transparantie buiten het gebouwsegment.

    Parameters:
    - image_rgb (np.ndarray): De originele RGB-afbeelding.
    - mask (np.ndarray): Een boolean-masker van het gebouwsegment.

    Returns:
    - rgba (np.ndarray): RGBA-afbeelding met het gebouw zichtbaar en de rest transparant.
    """
    huis_rgba = np.zeros((image_rgb.shape[0], image_rgb.shape[1], 4), dtype=np.uint8)
    huis_rgba[:, :, :3] = image_rgb
    huis_rgba[:, :, 3] = (mask * 255).astype(np.uint8)

    return huis_rgba


def genereer_segmenten(sam, image_rgb):
    """
    Voer automatische segmentatie uit op een afbeelding met behulp van SAM.

    Parameters:
    - sam: Een geïnitialiseerd SAM-model.
    - image_rgb (np.ndarray): De RGB-afbeelding waarop segmentatie wordt uitgevoerd.

    Returns:
    - resultaten (List[Dict]): Een lijst van segmentatieresultaten, elk met o.a. een 'segmentation'-masker.
    """
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator.generate(image_rgb)


def kleur_segmenten_op_grootte(resultaten, alpha_mask, image_rgb):
    """
    Kleur segmenten in de afbeelding op basis van hun grootte en locatie binnen het gebouw.

    Parameters:
    - resultaten (List[Dict]): Segmentatieresultaten van SAM met maskers.
    - alpha_mask (np.ndarray): Boolean-masker van het gebouw (True = gebouw).
    - image_rgb (np.ndarray): De afbeelding waarin gekleurd wordt.

    Returns:
    - image_rgb (np.ndarray): De afbeelding met gekleurde segmenten:
        * Groen voor kleine objecten (<10%)
        * Geel voor middelgrote objecten (10%-50%)
        * Rood voor grote objecten (>50%)
    """
    max_area = max(np.sum(m["segmentation"]) for m in resultaten)

    for m in resultaten:
        mask = m["segmentation"]
        area = np.sum(mask)

        # Segmenten buiten het gebouw negeren
        geldig_mask = mask & alpha_mask
        if np.sum(geldig_mask) < area * 0.5:
            continue

        # Kleur op basis van grootte
        if area < max_area * 0.1:
            kleur = [0, 255, 0]  # groen
        elif area < max_area * 0.5:
            kleur = [255, 255, 0]  # geel
        else:
            kleur = [255, 0, 0]  # rood

        for c in range(3):
            image_rgb[:, :, c][geldig_mask] = kleur[c]

    return image_rgb


def kleur_dak_blauw_als_geldig(gekleurd_rgb, dak_mask, gebouw_mask):
    """
    Kleur de dakgebieden blauw als het dakoppervlak tussen 10% en 70% van het gebouwoppervlak is.

    Parameters:
    - gekleurd_rgb: De al ingekleurde RGB-afbeelding (numpy-array)
    - dak_mask: Boolean numpy-masker van het dak
    - gebouw_mask: Boolean numpy-masker van het hele gebouw

    Returns:
    - aangepaste RGB-afbeelding
    """
    # Bepaal het aantal pixels in het dak en gebouw
    dak_area = np.sum(dak_mask)
    gebouw_area = np.sum(gebouw_mask)

    # Controleer of dakoppervlak tussen 10% en 70% van het gebouw ligt
    if gebouw_area == 0:
        return gekleurd_rgb  # veiligheidscheck

    verhouding = dak_area / gebouw_area
    if 0.1 <= verhouding <= 0.7:
        # Pas blauwe kleur toe op dakpixels
        blauw = [0, 0, 255]

        # Vind pixels binnen het dakmask die NIET al groen zijn
        groen_mask = (
                (gekleurd_rgb[:, :, 0] == 0) &
                (gekleurd_rgb[:, :, 1] == 255) &
                (gekleurd_rgb[:, :, 2] == 0)
        )

        # Alleen de dakpixels die niet groen zijn → blauw maken
        dak_alleen_zonder_groen = dak_mask & ~groen_mask

        for c in range(3):
            gekleurd_rgb[:, :, c][dak_alleen_zonder_groen] = blauw[c]

    return gekleurd_rgb


def verwijder_niet_rechthoekige_groene_segmenten(image_rgb, vorm_tolerantie=0.6):
    """
    Verwijdert groene segmenten uit een RGB-afbeelding als ze geen rechthoekachtige vorm hebben.
    :param image_rgb: Numpy-array van de afbeelding
    :param vorm_tolerantie: Hoe rechthoekig iets moet zijn (1 = perfecte rechthoek, 0.6 = tolerant)
    :return: opgeschoonde afbeelding
    """
    # Maak een binair masker van alle groene pixels (ramen/deuren)
    groen_masker = np.all(image_rgb == [0, 255, 0], axis=2).astype(np.uint8) * 255

    # Zoek contouren in het groene masker
    contours, _ = cv2.findContours(groen_masker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        rect_area = w * h

        if rect_area == 0:
            continue  # voorkom delen door nul

        verhouding = area / rect_area

        # Als het groene object niet rechthoekig genoeg is (zoals gras of bomen), verwijder het
        if verhouding < vorm_tolerantie:
            cv2.drawContours(image_rgb, [contour], -1, (255, 0, 0), thickness=cv2.FILLED)

    return image_rgb


def load_ground_truth_mask(txt_path, img_shape, class_id=3):
    """
    Laadt een ground-truth segmentatiemasker uit een YOLOv8 .txt-bestand.

    Parameters:
    - txt_path (str): Pad naar het .txt-bestand met genormaliseerde polygonen.
    - img_shape (tuple): Vorm van de afbeelding als (hoogte, breedte).
    - class_id (int): De klasse-ID die je wilt extraheren (bijv. 3 = raam/deur).

    Returns:
    - mask (np.ndarray): Boolean numpy-masker (True = pixels van opgegeven klasse).
    """
    # Maak leeg binair masker
    h, w = img_shape[:2]
    ground_truth_mask = np.zeros((h, w), dtype=np.uint8)

    # Lees polygonen uit YOLO .txt en vul ze in het masker
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if int(parts[0]) != class_id:
                continue

            coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
            coords[:, 0] *= w  # X → pixels
            coords[:, 1] *= h  # Y → pixels
            polygon = np.round(coords).astype(np.int32)
            cv2.fillPoly(ground_truth_mask, [polygon], 255)

    # Zet om naar boolean masker voor gebruik met IoU
    return ground_truth_mask > 0


def bereken_iou_en_print(sam_groen_mask, ground_truth_mask):
    """
    Berekent en print de Intersection over Union (IoU) tussen twee maskers.

    Parameters:
    - sam_groen_mask (np.ndarray): Boolean-masker van je SAM-output.
    - ground_truth_mask (np.ndarray): Boolean-masker van de YOLOv8 ground-truth.

    Output:
    - Print de IoU-score tussen SAM-segmentatie en grondwaarheid.
    """
    # Bepaal intersectie en unie tussen de twee maskers
    intersectie = np.logical_and(sam_groen_mask, ground_truth_mask).sum()
    unie = np.logical_or(sam_groen_mask, ground_truth_mask).sum()

    # Bereken IoU
    iou = intersectie / unie
    print(f"IoU-score tussen SAM-output en YOLO-ground-truth: {iou:.4f}")

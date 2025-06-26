from sorce.sam import *
from PIL import Image


def main(afbeelding):
    # === Parameters ===
    input_pad = f"Resultaten_met_Contouren/{afbeelding}"
    output_rgba_pad = f"Data_nest_uitgesneden_zonder_achtergrond/{afbeelding}"
    output_kleur_pad = f"Data_nest_uitgesneden_output/{afbeelding}"

    # === Model laden ===
    sam, predictor = laad_sam_model()

    # === Afbeelding laden en instellen voor SAM ===
    image, image_rgb = laad_afbeelding_en_prepareer(input_pad, predictor)
    h, w, _ = image.shape

    # === Punten genereren voor dak en gebouw ===
    dak_punten, dak_labels = genereer_punten(h, w, mode="dak")
    gebouw_punten, gebouw_labels = genereer_punten(h, w, mode="gebouw")

    # === Maskers voorspellen ===
    dak_mask = predict_mask(predictor, dak_punten, dak_labels)
    gebouw_mask = predict_mask(predictor, gebouw_punten, gebouw_labels)

    # === Toon maskers (voor debugging/controle) ===
    toon_masker(dak_mask, dak_punten, titel="Dakmasker")
    toon_masker(gebouw_mask, gebouw_punten, titel="Gebouwmasker")

    # === Transparante afbeelding van alleen het gebouw maken ===
    huis_rgba = maak_rgba_gebouw(image_rgb, gebouw_mask)
    Image.fromarray(huis_rgba).save(output_rgba_pad)  # Het gebouw zonder achtergrond opslaan

    # === Segmenten genereren (volledige afbeelding) ===
    resultaten = genereer_segmenten(sam, image_rgb)

    # === Alleen segmenten binnen het gebouw inkleuren op grootte ===
    alpha_mask = huis_rgba[:, :, 3] > 0

    gekleurd_rgb = kleur_segmenten_op_grootte(resultaten, alpha_mask, image_rgb)
    gekleurd_rgb = kleur_dak_blauw_als_geldig(gekleurd_rgb, dak_mask, gebouw_mask)
    gekleurd_rgb = verwijder_niet_rechthoekige_groene_segmenten(gekleurd_rgb)

    # === Eindresultaat tonen en opslaan ===
    plt.imshow(gekleurd_rgb)
    plt.title("Segmentatie gekleurd")
    plt.axis("off")
    plt.show()

    Image.fromarray(gekleurd_rgb).save(output_kleur_pad)  # De output opslaan

    # Stel in: paden en masker van jouw afbeelding
    txt_pad = "half_vrijstaand_type_e-4-_png.rf.43e530b98b2c051a8aa4a235fddc4957.txt"

    # Genereer ground-truth masker (alleen klasse 3 = raam/deur)
    gt_mask = load_ground_truth_mask(txt_pad, image.shape, class_id=3)

    # Genereer je eigen mask van SAM-segmentatie (alleen de groene delen)
    sam_mask = np.all(gekleurd_rgb == [0, 255, 0], axis=-1)

    # Bereken en print de IoU
    bereken_iou_en_print(sam_mask, gt_mask)


main("half_vrijstaand_type_e (4)_contours.png")

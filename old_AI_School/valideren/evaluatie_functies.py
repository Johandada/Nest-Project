"""
Dit bestand is gemaakt om deze functies te kunnen testen voor unittests in een notebook, 
en om ze aan te roepen in andere notebooks om te valideren.
"""


def calculate_iou(box_a, box_b):
    """
    Berekent de Intersection over Union (IoU) tussen twee bounding boxes.
    
    Parameters:
        box_a (list or tuple): Bounding box in [x, y, w, h]-formaat.
        box_b (list or tuple): Bounding box in [x, y, w, h]-formaat.
    
    Returns:
        float: IoU-waarde tussen 0.0 en 1.0.
    """
    # Zet om naar [x1, y1, x2, y2]
    x1_a, y1_a, w_a, h_a = box_a
    x2_a, y2_a = x1_a + w_a, y1_a + h_a

    x1_b, y1_b, w_b, h_b = box_b
    x2_b, y2_b = x1_b + w_b, y1_b + h_b

    # Bereken overlapcoördinaten
    x_left = max(x1_a, x1_b)
    y_top = max(y1_a, y1_b)
    x_right = min(x2_a, x2_b)
    y_bottom = min(y2_a, y2_b)

    # Check of er overlap is
    if x_right < x_left or y_bottom < y_top:
        return 0.0  # Geen overlap

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box_a_area = w_a * h_a
    box_b_area = w_b * h_b
    union_area = box_a_area + box_b_area - intersection_area

    return intersection_area / union_area


IOU_THRESHOLD = 0.5

def match_detections_to_gt(image, gt_annotations, detection_data):
    """
    Vergelijkt gedetecteerde objecten met ground-truth annotaties voor een afbeelding.
    
    Parameters:
        image (dict): Metadata van de afbeelding, bevat 'id' en 'file_name'.
        gt_annotations (list): Lijst met ground-truth annotaties in COCO-formaat.
        detection_data (list): Lijst met detectie-uitvoer, met bounding boxes en labels.
    
    Returns:
        dict: Informatie over het aantal grondwaarheden, detecties en correcte matches.
    """
    image_id = image["id"]
    file_name = image["file_name"]

    # Filter ground-truth bboxes met category_id = 4 (Raam)
    gt_boxes = [
        ann["bbox"]
        for ann in gt_annotations
        if ann["image_id"] == image_id and ann["category_id"] == 4
    ]

    # Filter detecties: alleen objecten met label 'Raam' of 'Deur'
    det_boxes = [
        det["bounding_box"]
        for det in detection_data
        if det["label"] in ["Raam", "Deur"]
    ]

    matched_gt = set()
    correct_detections = 0

    for det_box in det_boxes:
        for i, gt_box in enumerate(gt_boxes):
            if i in matched_gt:
                continue

            iou = calculate_iou(det_box, gt_box)

            if iou >= IOU_THRESHOLD:
                correct_detections += 1
                matched_gt.add(i)
                break  # Stop met zoeken zodra match is gevonden

    return {
        "file_name": file_name,
        "gt_count": len(gt_boxes),
        "det_count": len(det_boxes),
        "correct": correct_detections
    }
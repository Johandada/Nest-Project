from yolo_model.nestlogica import *


def test_genereer_gevel_en_venster_maskers():
    """
        Test of het masker correct ramen en gevel genereert op basis van dummy YOLO-output.
    """

    class Dummy_cls:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class Dummy_box:
        def __init__(self, cls_index, coords):
            self.cls = Dummy_cls(cls_index)
            self.xyxy = [coords]

    class Dummy_results:
        def __init__(self):
            self.names = ['facade', 'roof', 'tree', 'window']
            self.boxes = [Dummy_box(3, [20, 20, 40, 40])]
            self.masks = None

    dummy_img_size = (64, 64)
    dummy_results = Dummy_results()

    facade_mask, window_mask = genereer_gevel_en_venster_maskers(dummy_results, dummy_img_size)

    assert window_mask[30, 30] == True
    assert window_mask[10, 10] == False
    assert facade_mask.shape == dummy_img_size


def test_nestlocaties_wel_geschikt():
    """
        Test of er 5 geschikte nestlocaties gevonden worden bij een afbeelding waar wél ruimte is.
    """
    img_path = "Test_data/rijwoning (8).png"
    image = Image.open(img_path)
    species = "huismus"

    icon_paths = {
        "huismus": "foto's/huismus.png",
        "gierzwaluw": "foto's/gierzwaluw.png",
        "vleermuis": "foto's/vleermuis.png"
    }

    model_path = "../getrainde modellen/yolov8n_nest_50epochs.pt"
    _, locaties = analyseer_nestlocaties(image, species, model_path, icon_paths)

    assert len(locaties) == 5


def test_nestlocaties_niet_geschikt():
    """
        Test of er geen nestlocaties worden gevonden wanneer de gevel ongeschikt is.
    """
    img_path = "Test_data/rijwoning (1).png"
    image = Image.open(img_path)
    species = "huismus"

    icon_paths = {
        "huismus": "foto's/huismus.png",
        "gierzwaluw": "foto's/gierzwaluw.png",
        "vleermuis": "foto's/vleermuis.png"
    }

    model_path = "../getrainde modellen/yolov8n_nest_50epochs.pt"
    _, locaties = analyseer_nestlocaties(image, species, model_path, icon_paths)

    assert len(locaties) == 0


def test_selecteer_nestlocaties():
    """
        Test of de geselecteerde nestlocaties minimaal 3 meter uit elkaar liggen
        en maximaal 5 locaties worden gekozen.
    """
    mask = np.zeros((100, 100), dtype=bool)
    mask[10, 10] = True
    mask[20, 80] = True
    mask[50, 50] = True
    mask[70, 20] = True
    mask[80, 80] = True
    mask[85, 85] = True  # te dicht bij (80,80)

    locaties = selecteer_nestlocaties(mask)

    assert len(locaties) <= 5
    for i, (y1, x1) in enumerate(locaties):
        for y2, x2 in locaties[i + 1:]:
            assert np.hypot(x1 - x2, y1 - y2) >= int(3.0 / PIXEL_SCALE)

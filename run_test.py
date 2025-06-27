from nestplaatsing import (
    load_model,
    load_image,
    detect_objects,
    create_masks,
    create_allowed_mask,
    select_nest_locations,
    visualize_nests,
    print_results
)

# 📂 Paden instellen
IMAGE_PATH = r"C:\Users\moham\Nest-Project\nest_data\test\images\test_3.jpg"
MODEL_PATH = r"C:\Users\moham\Nest-Project\yolov8n_nest_50epochs.pt"
ICON_PATHS = {
    'huismus':     r"C:\Users\moham\Nest-Project\huismus.png",
    'gierzwaluw':  r"C:\Users\moham\Nest-Project\gierzwaluw.png",
    'vleermuis':   r"C:\Users\moham\Nest-Project\vleermuis.png"
}

# 📐 Schaalinstellingen
PIXEL_SCALE = 0.0175  # meter per pixel

# 🐦 Regels per dier
rules = {
    'huismus':     {'min_h': 3, 'max_h': 10},
    'gierzwaluw':  {'min_h': 6, 'max_h': 40},
    'vleermuis':   {'min_h': 3, 'max_h': 50}
}

# 👤 Gebruiker kiest diersoort
species = input("Kies een diersoort (huismus, gierzwaluw, vleermuis): ").strip().lower()
if species not in rules:
    raise ValueError("Ongeldige invoer. Kies uit: huismus, gierzwaluw, vleermuis.")

# 🚀 Verwerking starten
image = load_image(IMAGE_PATH)
model = load_model(MODEL_PATH)
results = detect_objects(model, IMAGE_PATH)

facade_mask, window_mask = create_masks(results, image.size[::-1])
allowed_mask = create_allowed_mask(
    facade_mask, window_mask, image.size[::-1],
    PIXEL_SCALE, rules[species]['min_h'], rules[species]['max_h']
)

selected_nests = select_nest_locations(allowed_mask, PIXEL_SCALE)

# 🎯 Visualisatie & resultaten
visualize_nests(image, allowed_mask, selected_nests, ICON_PATHS[species], species, PIXEL_SCALE)
print_results(selected_nests, species, PIXEL_SCALE)

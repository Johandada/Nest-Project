from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import os

# 🔧 Pad naar jouw getrainde model (.pt-bestand)
model_path = r"C:\Users\moham\Nest-Project\yolov8n_nest_50epochs.pt"
model = YOLO(model_path)

# 📸 Specifieke afbeelding om te testen
image_path = r"C:\Users\moham\Nest-Project\AI_generated_facades\facade.jpg"

# 📁 Map om resultaat op te slaan
output_folder = r"C:\Users\moham\CLONED\Nest-Project\Nest-Project\output_resultaten"
os.makedirs(output_folder, exist_ok=True)

# 🔍 Voorspelling uitvoeren
results = model.predict(source=image_path, save=False)

# 💾 Resultaat opslaan
filename = os.path.basename(image_path)
save_path = os.path.join(output_folder, f"result_{filename}")
results[0].save(filename=save_path)

# 📸 Resultaat weergeven
img = Image.open(save_path)
plt.imshow(img)
plt.axis('off')
plt.title(f"Resultaat: {filename}")
plt.show()

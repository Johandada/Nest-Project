import streamlit as st
from PIL import Image
from model_predict import run_model
from nestlogica import analyseer_nestlocaties

col1, col2 = st.columns([1, 1])

with col1:
    st.image("nest_logo.png", width=150)

with col2:
    st.image("hu_logo.png", width=100)
    st.markdown("   ")

st.title("YOLOv8 Nest Model Dashboard")

# 🐦 Soortkeuze
selected_species = st.selectbox(
    "Kies een diersoort voor nestanalyse:",
    ["huismus", "gierzwaluw", "vleermuis"]
)

uploaded_file = st.file_uploader("Upload een afbeelding", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Geüploade afbeelding", use_container_width=True)

    if st.button("Voorspel"):
        with st.spinner("Model wordt toegepast..."):
            result = run_model(image)
        st.success("Analyse voltooid!")
        st.image(result["image_with_boxes"], caption="📷 Resultaat met bounding boxes", use_container_width=True)
        st.write("🔎 Aantal objecten:", result["n_boxes"])
        st.write("🔖 Labels:", result["labels"])
        st.write("📊 Confidences:", result["confidences"])

    if st.button("Start nestanalyse"):
        iconen = {
            'huismus': 'huismus.png',
            'gierzwaluw': 'gierzwaluw.png',
            'vleermuis': 'vleermuis.png'
        }
        with st.spinner("Nestlocaties worden berekend..."):
            fig, locaties = analyseer_nestlocaties(image, selected_species, "yolov8n_nest_50epochs.pt", iconen)
        st.pyplot(fig)

        st.write(f"### 📍 Nestlocaties voor {selected_species}")
        for i, loc in enumerate(locaties, 1):
            st.write(f"Nest {i}: x={loc['x']}, y={loc['y']} → hoogte ≈ {loc['hoogte_m']} m")

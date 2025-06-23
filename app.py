import streamlit as st
from PIL import Image
from model_predict import run_model

st.title("YOLOv8 Nest Model Dashboard")

uploaded_file = st.file_uploader("Upload een afbeelding", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Geüploade afbeelding", use_column_width=True)

    if st.button("Voorspel"):
        with st.spinner("Model wordt toegepast..."):
            result = run_model(image)
        st.success("Analyse voltooid!")

        st.image(result["image_with_boxes"], caption="📷 Resultaat met bounding boxes", use_column_width=True)
        st.write("🔎 Aantal objecten:", result["n_boxes"])
        st.write("🔖 Labels:", result["labels"])
        st.write("📊 Confidences:", result["confidences"])


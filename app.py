import streamlit as st
from PIL import Image
from model_predict import run_model
col1, col2 = st.columns([1, 1])

from Logica_functie import analyseer

with col1:
    st.image("nest_logo.png", width=150)

with col2:
    st.image("hu_logo.png", width=100)
    st.markdown("   ")
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

    if st.button("Start nestanalyse"):
        from Logica_functie import analyseer

        fig = analyseer(image, "yolov8n_nest_50epochs.pt")
        st.pyplot(fig)




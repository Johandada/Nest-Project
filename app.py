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

st.markdown("""
### Welkom bij het Nest Model Dashboard
Deze applicatie helpt je geschikte locaties voor nestkasten te vinden op basis van een geüploade foto van een gebouw.

1. Kies een diersoort.
2. Upload een gevelafbeelding.
3. Klik op **Voer analyse uit** om te zien waar nestkasten geplaatst kunnen worden.
""")

selected_species = st.selectbox(
    "Kies een diersoort voor nestanalyse:",
    ["huismus", "gierzwaluw", "vleermuis"],
    help="Selecteer de soort waarvoor je nestlocaties wilt analyseren"
)

uploaded_file = st.file_uploader("Upload een afbeelding", type=["jpg", "jpeg", "png"],
    help="Upload een gevelafbeelding met zichtbare ramen en muur"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Geüploade afbeelding", use_container_width=True)

    if st.button("Voer analyse uit"):
        iconen = {
            'huismus': 'huismus.png',
            'gierzwaluw': 'gierzwaluw.png',
            'vleermuis': 'vleermuis.png'
        }
        with st.spinner("Nestlocaties worden berekend..."):
            fig, locaties = analyseer_nestlocaties(image, selected_species, "yolov8n_nest_50epochs.pt", iconen)
        st.pyplot(fig)

        st.markdown(f"### Plaats de nestkasten voor **{selected_species}** op de volgende locaties:")
        for i, loc in enumerate(locaties, 1):
            st.write(f"Nest {i}: x={loc['x']}, y={loc['y']} → hoogte ≈ {loc['hoogte_m']} m")

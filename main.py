import streamlit as st
from PIL import Image
from Modellen.nestlogica import analyseer_nestlocaties



def toon_header():
    """
        Toont de header van de Streamlit-app met projectlogo’s en introductietekst.
    """
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("foto's/nest_logo.png", width=150)
    with col2:
        st.image("foto's/hu_logo.png", width=100)
        st.markdown("   ")

    st.title("YOLOv8 Nest Model Dashboard")
    st.markdown("""
    ### Welkom bij het Nest Model Dashboard
    Deze applicatie helpt je geschikte locaties voor nestkasten te vinden op basis van een geüploade foto van een gebouw.

    1. Kies een diersoort.
    2. Upload een gevelafbeelding.
    3. Klik op **Voer analyse uit** om te zien waar nestkasten geplaatst kunnen worden.
    """)


def selecteer_diersoort():
    """
        Laat de gebruiker een diersoort selecteren voor nestkastanalyse.
        :return: geselecteerde diersoort (str)
    """
    return st.selectbox(
        "Kies een diersoort voor nestanalyse:",
        ["huismus", "gierzwaluw", "vleermuis"],
        help="Selecteer de soort waarvoor je nestlocaties wilt analyseren"
    )


def upload_afbeelding():
    """
        Laat de gebruiker een gevelafbeelding uploaden.
        :return: geüploade afbeelding als bestand
    """
    return st.file_uploader(
        "Upload een afbeelding",
        type=["jpg", "jpeg", "png"],
        help="Upload een gevelafbeelding met zichtbare ramen en muur"
    )


def voer_nestkast_analyse_uit(image, species):
    """
        Voert de analyse uit op basis van de geüploade afbeelding en geselecteerde diersoort.
        Toont het resultaat met een figuur en bijbehorende locaties.

        :param image: PIL-afbeelding die door gebruiker is geüpload
        :param species: geselecteerde diersoort (str)
    """
    modelpad = "Modellen/getrainde modellen/yolov8n_nest_50epochs.pt"
    iconen = {
        'huismus': "foto's/huismus.png",
        'gierzwaluw': "foto's/gierzwaluw.png",
        'vleermuis': "foto's/vleermuis.png"
    }

    with st.spinner("Nestlocaties worden berekend..."):
        fig, locaties = analyseer_nestlocaties(image, species, modelpad, iconen)

    # Visualisatie van de output
    st.pyplot(fig)

    # Toon coördinaten van de voorgestelde nestkasten
    st.markdown(f"### Plaats de nestkasten voor **{species}** op de volgende locaties:")
    for i, loc in enumerate(locaties, 1):
        st.write(f"Nest {i}: x={loc['x']}, y={loc['y']} → hoogte ≈ {loc['hoogte_m']} m")


def main():
    """
        Hoofdfunctie van de Streamlit-app.
        Regelt gebruikersinteractie: header tonen, soort kiezen, afbeelding uploaden en analyse uitvoeren.
    """
    toon_header()
    species = selecteer_diersoort()
    uploaded_file = upload_afbeelding()

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Geüploade afbeelding", use_container_width=True)

        if st.button("Voer analyse uit"):
            voer_nestkast_analyse_uit(image, species)


if __name__ == "__main__":
    main()

    
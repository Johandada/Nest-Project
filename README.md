Nestkastdetectie met AI

Automatische herkenning van geschikte nestkastlocaties op architectonische geveltekeningen met behulp van beeldherkenning (OpenCV + Segment Anything + PyTorch).

Projectbeschrijving

Door renovatie of sloop verdwijnen natuurlijke nestplekken voor vogels zoals de huismus, gierzwaluw en gewone dwergvleermuis. Dit project ontwikkelt een AI-model dat automatisch geschikte posities voor nestkasten detecteert op gevelaanzichten.

Samenwerking met: Nest NatuurinclusiefStudenten: Johan Dada, Mustafa El Yusuf, Ali Albonaser, Khalid AlkahloutBegeleiding: Ingrid Sloots (Nest)

Installatie

Voorwaarden

Python 3.10

Miniconda of Anaconda

1. Conda-omgeving instellen

Gebruik het meegeleverde environment.yml:

conda env create -f environment.yml
conda activate environment

2. Segment Anything installeren

We gebruiken het model van Meta AI: facebookresearch/segment-anything

git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .

3. Model downloaden

Download het voorgetrainde model en plaats het in de hoofdmap van het project:

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

🚀 Uitvoeren van de code

Activeer de omgeving:

conda activate environment

Start het notebook:

jupyter notebook sam.ipynb

Stap voor stap:

Laadt een geveltekening (Data_nest_uitgesneden/)

Converteert deze naar RGB

Genereert invoerpunten rond het midden

Roept het Segment Anything model aan

Toont de voorspelde maskers + nestkastlocaties

📊 Voorbeeldvisualisatie

plt.imshow(beste_mask)
plt.scatter(punten[:, 0], punten[:, 1], color="blue", s=40, marker="o", label="Invoerpunt")
plt.legend()

📂 Projectstructuur

Nest-Project/
├── Data_nest_uitgesneden/        # Invoergegevens (geveltekeningen)
├── sam.ipynb                     # Hoofdanalyse notebook
├── environment.yml              # Conda dependencies
├── sam_vit_h_4b8939.pth         # Segment Anything model
├── README.md                    # Deze instructie

📄 Licentie

Dit project is ontwikkeld als onderdeel van een onderzoeksproject aan de Hogeschool Utrecht.
Gebruik van tekeningen/data van Nest Natuurinclusief is alleen toegestaan binnen het project en niet voor externe verspreiding.

🙌 Contact

Voor vragen over het project:

Johan Dada (contactpersoon via GitHub of Teams)

Of neem contact op met Nest Natuurinclusief

"De natuur een stem aan tafel geven" — Nest Natuurinclusief

# Nestkastdetectie met YOLOv8

Automatische herkenning van geschikte nestlocaties op gevelafbeeldingen met behulp van objectdetectie (YOLOv8) en logica voor ecologische geschiktheid. Het systeem draait in een Streamlit-dashboard.

---

## Projectbeschrijving

De applicatie detecteert automatisch geschikte nestlocaties voor soorten zoals de **huismus**, **gierzwaluw** en **gewone dwergvleermuis**. Het systeem is gebouwd met een getraind YOLOv8-model, dat gevels en ramen detecteert, en logica toepast op basis van afstand en hoogtecriteria.

---

## Installatie

### Voorwaarden

- Python 3.10  
- Miniconda of Anaconda

### 1. Conda-omgeving instellen

Gebruik het meegeleverde `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate nestdetectie
```

### 2. Bestandenstructuur

```
Nest-Project/
├── main.py                    # Streamlit-dashboard
├── model_predict.py          # Testscript voor model
├── yolo_model/
│   └── nestlogica.py         # Detectie + filtering
├── Modellen/
│   └── yolo/
│       ├── yolov8n_nest_50epochs.pt
│       ├── yolov8m_nest_50epochs.pt
│       └── info/
│           └── technische eisen.docx
├── notebooks/
├── data/
├── icons/
└── environment.yaml
```

### 3. Model downloaden

Plaats het gewenste YOLOv8-modelbestand (bijv. `yolov8n_nest_50epochs.pt`) in de juiste map.

---

## Uitvoeren van de code

### 1. Activeer de omgeving

```bash
conda activate nestdetectie
```

### 2. Start de app

```bash
streamlit run main.py
```

### 3. Stap voor stap

- Kies een soort en upload een gevelafbeelding.
- Het model detecteert gevels en ramen.
- Logica filtert geschikte zones (afstand tot rand, ramen, hoogte).
- Maximaal 5 locaties worden gekozen (≥3 m uit elkaar).
- Visualisatie met mask en icoontjes

---

## Technische eisen

De volledige technische architectuur, pipelinebeschrijving en logica zijn gedocumenteerd in:

📄 `Modellen/info/technische eisen.docx`

**Bevat o.a.:**
- Analysepipeline van afbeelding tot nestlocatie
- Logica en filtering op basis van ecologische regels
- Alternatieve modellen (SAM, OpenCV)
- Onderhoud en uitbreidingsadvies

---

## Projectstructuur

```
Nest-Project/
├── data/                     # Testdata
├── notebooks/                # Experimenten & modeltraining
├── yolo_model/              # Segmentatie + logica
├── Modellen/                # Getrainde modellen
├── main.py                  # Streamlit UI
├── model_predict.py         # Model testscript
├── environment.yaml         # Conda dependencies
└── README.md                # Projectdocumentatie
```

---

## Licentie

Dit project is ontwikkeld als onderdeel van een onderzoeksproject aan de **Hogeschool Utrecht**. Gebruik van tekeningen/data van **Nest Natuurinclusief** is alleen toegestaan binnen het project en niet voor externe verspreiding.

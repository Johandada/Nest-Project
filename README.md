#  **Nestkastdetectie met AI**

Automatische herkenning van geschikte nestkastlocaties op architectonische geveltekeningen met behulp van beeldherkenning (OpenCV + Segment Anything).

---

##  **Projectbeschrijving**

Door renovatie of sloop verdwijnen natuurlijke nestplekken voor vogels zoals de **huismus**, **gierzwaluw** en **gewone dwergvleermuis**. Dit project ontwikkelt een AI-model dat **automatisch geschikte posities voor nestkasten** detecteert op gevelaanzichten.


---

##  **Installatie**

### **Voorwaarden**

* Python **3.10**
* **Miniconda** of **Anaconda**

### **1. Conda-omgeving instellen**

Gebruik het meegeleverde `environment.yml`:

```bash
conda env create -f environment.yml
conda activate sam-nestkast
```

### **2. Segment Anything installeren**

We gebruiken het model van Meta AI: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)


### **3. Model downloaden**

Download het voorgetrainde model en plaats het in de hoofdmap van het project:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

---

##  **Uitvoeren van de code**

### **1. Activeer de omgeving:**

```bash
conda activate sam-nestkast
```

### **2. Start het notebook:**

```bash
jupyter notebook sam.ipynb
```

### **3. Stap voor stap:**

* Laadt een geveltekening (`Data_nest_uitgesneden/`)
* Converteert deze naar RGB
* Genereert invoerpunten rond het midden
* Roept het Segment Anything model aan
* Toont de voorspelde maskers + nestkastlocaties

---

##  **Projectstructuur**

```
Nest-Project/
├── Data_nest_uitgesneden        # De data
├── sam.ipynb                    # Hoofdanalyse notebook
├── environment.yml              # Conda dependencies
├── sam_vit_h_4b8939.pth         # Segment Anything model
├── README.md                    # Projectdocumentatie
```

---

##  **Licentie**

Dit project is ontwikkeld als onderdeel van een onderzoeksproject aan de **Hogeschool Utrecht**.
Gebruik van tekeningen/data van **Nest Natuurinclusief** is alleen toegestaan binnen het project en niet voor externe verspreiding.

---

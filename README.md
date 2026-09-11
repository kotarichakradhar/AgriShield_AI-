# 🌿 AgriShield AI

> **Precision Crop Disease Diagnostics** · Computer Vision · IBM Granite Reasoning  
> Supporting **UN SDG 2** (Zero Hunger) and **UN SDG 15** (Life on Land)

---

## Architecture Overview

```
AgriShield_AI/
├── app.py              ← Streamlit UI (3 tabs: Diagnostics · Weather · Conservation)
├── vision_engine.py    ← CV/DL pipeline (chlorosis, lesions, severity, MobileNetV3)
├── llm_engine.py       ← Multimodal reasoning (IBM Granite / rule-based fallback)
├── requirements.txt    ← Pinned production dependencies
└── .env.example        ← Environment variable template
```

### Pipeline

```
Image Upload (JPG/PNG)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              vision_engine.py                        │
│  RGB → HSV → Chlorosis %                            │
│  Canny edge → Contour detection → Lesion density    │
│  Composite Severity Score (0–100%)                  │
│  [Optional] MobileNetV3 deep feature extraction     │
└────────────────────┬────────────────────────────────┘
                     │  VisionReport
                     ▼
┌─────────────────────────────────────────────────────┐
│              llm_engine.py                           │
│  + Crop metadata (type, growth stage, region)       │
│  + Weather telemetry (humidity, rainfall, temp)     │
│  → IBM Granite structured prompt (RAG-style)        │
│  → Fallback: deterministic disease KB               │
└────────────────────┬────────────────────────────────┘
                     │  DiagnosticReport
                     ▼
┌─────────────────────────────────────────────────────┐
│                  app.py (Streamlit)                  │
│  Tab 1: Image Diagnostics + Annotated Image         │
│  Tab 2: Weather Telemetry + Disease Risk Gauges     │
│  Tab 3: SDG 2/15 Conservation Impact Dashboard     │
└─────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Install dependencies

```bash
cd AgriShield_AI
pip install -r requirements.txt
```

For **GPU acceleration** (CUDA 12.1):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. (Optional) Configure IBM Granite

```bash
cp .env.example .env
# Edit .env and set GRANITE_API_KEY
```

Without an API key, the app uses the built-in rule-based inference engine automatically.

### 3. Run

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Features

| Feature | Technology |
|---|---|
| Chlorosis / Yellowing Detection | OpenCV HSV colour space analysis |
| Necrotic Lesion Density | Canny edge detection + contour analysis |
| Composite Severity Scoring | Weighted (0–100%) composite metric |
| Deep Feature Extraction | MobileNetV3-Small backbone (optional) |
| Disease Diagnosis | IBM Granite 3-8B / rule-based KB fallback |
| Organic Interventions | Neem, bio-fungicides, PGPR |
| Biological Biocontrol | Trichoderma, Bacillus subtilis, Pseudomonas |
| SDG 2 Alignment | Yield loss prevention, smallholder food security |
| SDG 15 Alignment | Zero synthetic toxins, pollinator & microbiome protection |
| Downloadable Report | Full Markdown diagnostic report |

---

## Supported Crops

Tomato · Rice · Cotton · Maize · Wheat · Soybean · Potato  
*(extensible via the `_DISEASE_KB` dictionary in `llm_engine.py`)*

---

## SDG Commitment

- **SDG 2 — Zero Hunger**: Early disease detection prevents 30–60% yield losses in smallholder farms.
- **SDG 15 — Life on Land**: All recommendations are strictly organic/biological. Neonicotinoids, organophosphates, persistent fungicides, soil fumigants, and synthetic herbicides are **permanently banned**.

---

## License

MIT — Free to use, modify, and distribute with attribution.

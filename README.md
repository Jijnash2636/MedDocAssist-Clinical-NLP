# MedDocAssist: Multimodal Clinical NLP System

## 🧠 Overview

MedDocAssist is an AI-powered clinical NLP system designed to automate medical documentation using multimodal inputs (text, audio, and images). The system integrates domain-specific transformer models to perform Named Entity Recognition (NER), ICD-10 coding, summarization, and drug-drug interaction detection.

This project focuses on building a safe, interpretable, and scalable clinical decision support system (CDSS) with human-in-the-loop validation.

---

## 🚀 Key Features

- ✅ Multimodal Input Processing (Text, Audio, Image)
- ✅ Clinical NER using BioBERT
- ✅ Abstractive Summarization using Flan-T5
- ✅ Semantic ICD-10 Coding using PubMedBERT
- ✅ PHI De-identification (HIPAA-aware)
- ✅ Drug-Drug Interaction Detection (CDSS)
- ✅ Confidence-Gated Human-in-the-Loop Validation
- ✅ Structured JSON Output for EHR integration

---

## 🧱 Project Structure

```
MedDocAssist-Clinical-NLP/
├── src/                   # Core modules
│   ├── main.py            # ClinicalNoteAI pipeline
│   ├── ner_biobert.py     # BioBERT NER
│   ├── summarizer.py      # Flan-T5 summarization
│   ├── icd_mapper.py      # ICD-10 mapping (34 codes)
│   ├── drug_interaction.py  # Drug interaction CDSS
│   ├── multimodal_input.py  # Audio/Image processing
│   └── __init__.py
│
├── results/               # IEEE figures (600 DPI)
│   ├── figure1_architecture_ieee.png
│   ├── figure2_normalization.png
│   ├── figure3_professional.png
│   ├── figure4_icd_mapping.png
│   ├── figure5_performance.png
│   └── figure6_confidence_gate.png
│
├── notebooks/             # Demo notebooks
│   └── demo.ipynb
│
├── app.py                # Full-featured UI
├── app_simple.py         # Lightweight UI (recommended)
├── evaluate.py           # Evaluation scripts
├── test_system.py        # Testing suite
├── requirements.txt      # Python dependencies
├── references.txt        # IEEE citations (23 refs)
└── README.md             # This file
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Jijnash2636/MedDocAssist-Clinical-NLP.git
cd MedDocAssist-Clinical-NLP
pip install -r requirements.txt
```

---

## ▶️ Usage

**Run the application:**

```bash
python app_simple.py
```

Then open your browser:
- **URL**: http://127.0.0.1:7864/

**Using the system:**
1. Select input type: Text / Audio / Image
2. Enter clinical data or upload file
3. Click "Analyze"
4. View results in each section

---

## 📊 Results

| Metric | Score |
|--------|-------|
| BioBERT NER F1-score | 0.87 (+15% over baseline) |
| ICD-10 Mapping Precision@3 | 0.89 |
| Summarization Compression | 42% |

---

## 🧠 System Architecture

The system follows a three-stage pipeline:

1. **Multimodal Ingestion** - Text, Audio (Whisper), Image (Tesseract)
2. **Pre-processing** - Text Normalization + PHI De-identification
3. **Clinical Intelligence Engine** - NER, Summarization, ICD Mapping, Drug Interaction

---

## 🔒 Safety Mechanisms

- **Confidence threshold** (C < 0.75 → human validation)
- **PHI masking** for privacy compliance
- **Drug interaction alerts** for risk prevention
- **Explainability** using SHAP

---

## 🧪 Evaluation

```bash
python evaluate.py
python test_system.py
```

---

## 📚 References

See `references.txt` for full IEEE citations (23 references).

---

## 👨‍💻 Authors

- **Jijnash Kumar**
- **Avinash K**
- **Priyadharsini C**

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgments

- BioBERT: Lee et al., Bioinformatics 2020
- Flan-T5: Chung et al., JMLR 2024
- PubMedBERT: Gu et al., ACM TOCH 2021

---

*For research and educational purposes. For clinical use, ensure compliance with local regulations.*

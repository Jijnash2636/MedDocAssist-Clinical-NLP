# MedDocAssist: Multimodal Clinical NLP System

## 🧠 Overview

MedDocAssist is an AI-powered clinical NLP system designed to automate medical documentation using multimodal inputs (text, audio, and images). The system integrates domain-specific transformer models to perform Named Entity Recognition (NER), ICD-10 coding, summarization, and drug-drug interaction detection.

This project focuses on building a safe, interpretable, and scalable clinical decision support system (CDSS) with human-in-the-loop validation and academic honesty in all outputs.

---

## 🚀 Key Features

- ✅ Explicit Mode-Based Routing (Text / Audio / Image isolation)
- ✅ Real OCR Extraction (Tesseract with auto-detection + preprocessing)
- ✅ Template Document Detection (distinguishes forms from real patient records)
- ✅ Whisper Audio Transcription (tiny model with labeled fallback)
- ✅ Clinical NER using BioBERT
- ✅ Abstractive Summarization using Flan-T5
- ✅ Semantic ICD-10 Coding using PubMedBERT
- ✅ PHI De-identification (HIPAA-aware)
- ✅ Drug-Drug Interaction Detection (CDSS)
- ✅ Wellness Note Recognition (healthy inputs handled professionally)
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
│   ├── multimodal_input.py  # Audio/Image/Video processing with template detection
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
├── samples/               # Test samples
│   ├── sample_audio.wav
│   └── sample_prescription.png
│
├── demo_live.py          # Main demo with mode-based routing
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

### Optional External Dependencies
- **Tesseract OCR** (for image processing): https://github.com/UB-Mannheim/tesseract/wiki
  - System auto-detects default install paths. Custom path can be configured in `src/multimodal_input.py`.
- **FFmpeg** (for audio processing): Required for Whisper transcription.

---

## ▶️ Usage

**Run the application:**

```bash
python demo_live.py
```

Then open your browser:
- **URL**: http://127.0.0.1:7864/

**Using the system:**
1. **Select Input Mode**: Text / Audio / Image (inputs are strictly isolated)
2. **Enter clinical data** or upload file
3. Click "Analyze"
4. View results: Raw extraction, Normalization, PHI removal, NER, ICD mapping, Summary, Drug check, Confidence score, JSON output

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

1. **Multimodal Ingestion** - Text, Audio (Whisper), Image (Tesseract + OCR preprocessing)
2. **Pre-processing** - Text Normalization + PHI De-identification + Template Detection (Image)
3. **Clinical Intelligence Engine** - NER, Summarization, ICD Mapping, Drug Interaction
4. **Validation Gate** - Confidence scoring + human review flagging

---

## 🔒 Safety & Academic Honesty

- **Explicit fallback labeling**: Simulated outputs are clearly marked (e.g., `[SIMULATED AUDIO TRANSCRIPT]`)
- **Template document detection**: Image module distinguishes clinical forms from real patient records
- **Confidence threshold** (C < 0.75 → human validation)
- **PHI masking** for privacy compliance
- **Drug interaction alerts** for risk prevention
- **Explainability** using SHAP
- **No hallucinated patient data** in error states

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
- Whisper: Radford et al., OpenAI 2022
- Tesseract-OCR: Google/UB Mannheim

---

*For research and educational purposes. For clinical use, ensure compliance with local regulations.*

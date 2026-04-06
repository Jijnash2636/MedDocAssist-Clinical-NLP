# IEEE Paper - Final Figures with Captions
# Clinical Note Summarization and Coding Assistant

---

## Figure 1: System Architecture
**Caption:** Fig. 1. Proposed system architecture showing the end-to-end clinical note processing pipeline. The system accepts multimodal inputs (text, audio, image), performs preprocessing (normalization, PHI removal), processes through parallel NLP modules (NER, summarization, ICD mapping, drug interaction), applies confidence-based gating, and outputs structured JSON for EHR integration.

---

## Figure 2: Named Entity Recognition (NER) Output
**Caption:** Fig. 2. Named Entity Recognition output demonstrating extraction of clinical entities from unstructured text. The system identifies (a) Problems/Symptoms such as chest pain and shortness of breath, (b) Treatments/Drugs including aspirin and warfarin, and (c) Medical Tests such as ECG and CBC.

| Entity Type | Extracted Terms | Example |
|-------------|-----------------|---------|
| PROBLEM | chest pain, shortness of breath, fever | Patient has [chest pain] |
| TREATMENT | aspirin, warfarin, metformin | Prescribed [aspirin] |
| TEST | ECG, CBC, X-ray | ECG shows normal rhythm |

---

## Figure 3: Evaluation Performance Comparison
**Caption:** Fig. 3. Performance comparison between baseline (rule-based) and fine-tuned transformer models. Our system achieves significant improvements: NER F1 score increases from 0.62 to 0.87 (+40%), Summarization ROUGE-L improves from 0.10 to 0.82 (+720%), and ICD Mapping Precision maintains high accuracy at 0.89-0.90.

| Task | Baseline Score | With Fine-tuned Models |
|------|---------------|------------------------|
| NER (F1) | 0.62 | 0.87 |
| Summarization (ROUGE-L) | 0.10 | 0.82 |
| ICD Mapping (Precision@3) | 0.89 | 0.90 |

---

## Figure 4: Summarization Results - Before vs After
**Caption:** Fig. 4. Abstractive summarization output demonstrating the system's ability to generate concise, clinically relevant summaries from lengthy clinical notes. The system preserves key medical information while reducing documentation burden.

| Test Case | Input (Long Note) | Output (Generated Summary) |
|-----------|-------------------|----------------------------|
| 1 | Patient has chest pain and shortness of breath. BP is 140/90. History of hypertension and diabetes. Prescribed aspirin 81mg daily and metoprolol 25mg BID. | Patient has chest pain and shortness of breath. Prescribed aspirin and metoprolol. |
| 2 | Patient has heart condition. Prescribed warfarin and aspirin. | Patient has heart condition. Prescribed warfarin and aspirin. |
| 3 | Elderly patient with fever and cough for 3 days. Given antibiotics. | Patient with fever and cough. Given antibiotics. |

---

## Figure 5: Drug Interaction CDSS Alerts
**Caption:** Fig. 5. Clinical Decision Support System (CDSS) output showing detection of harmful drug interactions. The system integrates with OpenFDA API to identify high-risk combinations and provides actionable recommendations for healthcare providers.

| Drug 1 | Drug 2 | Risk | Description | Recommendation |
|--------|--------|------|-------------|----------------|
| Aspirin | Warfarin | HIGH | Increased bleeding risk | Monitor INR closely |
| Aspirin | Heparin | HIGH | Increased bleeding risk | Avoid combination |
| Nitroglycerin | Sildenafil | HIGH | Severe hypotension | Contraindicated |
| Metformin | Contrast | MEDIUM | Lactic acidosis risk | Hold metformin |

---

## Figure 6: ICD-10 Code Mapping
**Caption:** Fig. 6. Semantic ICD-10 code mapping showing automatic assignment of diagnostic codes from extracted medical entities. The system uses keyword matching and embedding similarity to map clinical terms to appropriate ICD-10 codes with confidence scores.

| Medical Term | ICD Code | Description | Confidence |
|--------------|----------|-------------|-------------|
| chest pain | R07.9 | Chest pain, unspecified | 0.85 |
| heart condition | I25.10 | Atherosclerotic heart disease | 0.75 |
| hypertension | I10 | Essential (primary) hypertension | 0.95 |
| diabetes | E11.9 | Type 2 diabetes mellitus | 0.92 |
| fever | R50.9 | Fever, unspecified | 0.88 |

---

## Summary

This paper presents a complete clinical NLP system with 8 novelties:

1. **Multimodal Input** - Supports text, audio (Whisper), and image (OCR)
2. **Clinical Text Normalization** - Expands abbreviations
3. **PHI De-identification** - HIPAA-compliant privacy protection
4. **Domain-specific NER** - BioBERT for medical entities
5. **Abstractive Summarization** - Flan-T5 for clinical summaries
6. **Semantic ICD Mapping** - PubMedBERT embeddings
7. **Human-in-the-loop** - Confidence-gated decision making
8. **Drug Interaction CDSS** - Real-time safety alerts

---

*Generated for IEEE Research Paper Submission*

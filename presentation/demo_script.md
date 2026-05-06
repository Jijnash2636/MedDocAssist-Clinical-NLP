# MedDocAssist - Demo Script for Project Presentation

## Preparation Checklist
- [ ] Open `presentation/results_demo.html` in browser (Chrome/Edge)
- [ ] Have `app_simple.py` ready for live demo
- [ ] Ensure Python environment is active
- [ ] Test Gradio UI before presentation

---

## Demo Flow (5-7 Minutes)

### Part 1: Slideshow Presentation (2-3 min)

**Open**: `presentation/results_demo.html`
**Navigate**: Arrow keys or on-screen buttons

| Slide | Key Points to Mention |
|-------|----------------------|
| 1 - Title | "This is MedDocAssist, our AI-powered clinical NLP system" |
| 2 - Problem | "Doctors spend 50% time on paperwork - we automate this" |
| 3 - Pipeline | "5-stage pipeline from input to structured output" |
| 4 - NER | "BioBERT achieves 0.87 F1, 15% better than generic BERT" |
| 5 - PHI/ICD | "HIPAA-compliant PHI removal + ICD-10 mapping with 0.89 precision" |
| 6 - Summary | "42% compression ratio with clinical accuracy" |
| 7 - Gate | "Confidence threshold ensures human review for uncertain cases" |
| 8 - Future | "Ready for MIMIC-III fine-tuning and EHR integration" |

---

### Part 2: Live Demo (3-4 min)

**Step 1: Start the App**
```bash
"C:\Users\M.Jijnash kumar\anaconda3\envs\ai_env\python.exe" "D:\Projects\Med_doc\app_simple.py"
```

**Step 2: Open Browser**
- Navigate to: `http://localhost:7864/`

**Step 3: Demonstrate Text Input**
1. Select "Text" as input type
2. Paste this sample clinical note:
   ```
   Patient presents with chest pain and shortness of breath.
   History of hypertension and type 2 diabetes.
   Prescribed aspirin 81mg daily and metformin 500mg twice daily.
   BP: 140/90, HR: 88 bpm. ECG ordered.
   ```
3. Click "Analyze"
4. Walk through results:
   - **Normalized Text**: Show abbreviation expansion (BP → Blood Pressure)
   - **De-identified Text**: Show PHI redaction (if names present)
   - **NER Results**: Point out extracted symptoms, drugs, tests
   - **Summary**: Show compressed version
   - **ICD Codes**: Show mapped codes (R07.9, I10, E11.9)
   - **Drug Interactions**: Highlight any alerts

**Step 4: Show Confidence Gate**
- Explain: "Each prediction has a confidence score"
- "Scores below 0.75 are flagged for human review"

**Step 5: Optional - Audio/Image Demo**
- If time permits, demonstrate audio or image input

---

## Key Metrics to Emphasize

| Metric | Value | Significance |
|--------|-------|--------------|
| NER F1-Score | 0.87 | 15% improvement over baseline |
| ICD Precision@3 | 0.89 | High accuracy code mapping |
| Compression | 42% | Significant documentation reduction |
| PHI Detection | 100% | Full HIPAA compliance |
| Novel Contributions | 8 | Research-worthy innovations |

---

## Common Q&A Preparation

**Q: Why use synthetic data instead of MIMIC-III?**
A: "MIMIC-III requires credentialed access. We designed the system architecture to work with real clinical data when credentials are available."

**Q: What's the latency for processing?**
A: "Current implementation uses CPU inference. With GPU, we expect 3-5x speedup for real-time processing."

**Q: How does this compare to existing tools?**
A: "Unlike single-function tools, MedDocAssist integrates 8 capabilities in one pipeline: NER, summarization, ICD mapping, drug alerts, PHI removal, multimodal input, confidence gating, and normalization."

**Q: Can this be deployed in hospitals?**
A: "The modular design allows easy integration with existing EHR systems. Future work includes HIPAA-compliant deployment and fine-tuning on real clinical data."

---

## Closing Statement

"MedDocAssist demonstrates how AI can reduce physician burnout by automating clinical documentation. With 8 novel contributions and strong baseline results, this system is ready for real-world deployment with fine-tuning on clinical datasets."

---

## Backup: If Live Demo Fails

Have screenshots ready in `results/8k/` folder:
- Figure 1: Architecture
- Figure 3: NER & PHI
- Figure 5: Performance
- Figure 6: Confidence Gate

Show pre-run outputs from `results/evaluation_results.json`

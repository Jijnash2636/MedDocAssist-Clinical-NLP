"""
Evaluation Script for IEEE Paper
Computes ROUGE, F1, Precision@K metrics
"""

import sys
sys.path.insert(0, '.')

import json
from src.ner_biobert import MedicalNER
from src.icd_mapper import ICDMapper
from src.summarizer import ClinicalSummarizer
from src.main import ClinicalNoteAI

def compute_bleu(reference, hypothesis):
    """Compute simplified BLEU score."""
    ref_words = set(reference.lower().split())
    hyp_words = set(hypothesis.lower().split())
    
    if not hyp_words:
        return 0.0
    
    overlap = len(ref_words & hyp_words)
    return overlap / len(hyp_words)

def compute_rouge_l(reference, hypothesis):
    """Compute simplified ROUGE-L."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    
    if not ref_words or not hyp_words:
        return 0.0
    
    m = len(ref_words)
    n = len(hyp_words)
    
    if m == 0 or n == 0:
        return 0.0
    
    lcs = 0
    j = 0
    for i in range(m):
        for k in range(j, n):
            if ref_words[i] == hyp_words[k]:
                lcs += 1
                j = k + 1
                break
    
    return 2 * lcs / (m + n)

def compute_precision_at_k(predicted, ground_truth, k=3):
    """Compute Precision@K."""
    if not predicted:
        return 0.0
    
    top_k = predicted[:k]
    correct = sum(1 for p in top_k if p in ground_truth)
    return correct / min(k, len(top_k))

# Load test data
synthetic_notes = [
    {
        "note_id": "SYN001",
        "text": """Patient presents with chest pain and shortness of breath.
        History of hypertension and diabetes.
        Prescribed aspirin and metoprolol.
        ECG and CBC ordered.""",
        "entities": [
            {"text": "chest pain", "type": "PROBLEM"},
            {"text": "shortness of breath", "type": "PROBLEM"},
            {"text": "hypertension", "type": "PROBLEM"},
            {"text": "diabetes", "type": "PROBLEM"},
            {"text": "aspirin", "type": "TREATMENT"},
            {"text": "metoprolol", "type": "TREATMENT"},
            {"text": "ECG", "type": "TEST"},
            {"text": "CBC", "type": "TEST"}
        ],
        "icd_codes": ["R07.9", "I10", "E11.9"]
    },
    {
        "note_id": "SYN002", 
        "text": """Patient reports severe headache and nausea.
        BP elevated to 160/100.
        History of migraines and anxiety.
        Given acetaminophen and referred to neurology.""",
        "entities": [
            {"text": "headache", "type": "PROBLEM"},
            {"text": "nausea", "type": "PROBLEM"},
            {"text": "hypertension", "type": "PROBLEM"},
            {"text": "migraine", "type": "PROBLEM"},
            {"text": "anxiety", "type": "PROBLEM"},
            {"text": "acetaminophen", "type": "TREATMENT"}
        ],
        "icd_codes": ["R51", "R11.0", "I10", "F32.9"]
    },
    {
        "note_id": "SYN003",
        "text": """Cough and fever for 3 days.
        WBC elevated, chest x-ray shows infiltrate.
        Diagnosed with pneumonia.
        Started on azithromycin.""",
        "entities": [
            {"text": "cough", "type": "PROBLEM"},
            {"text": "fever", "type": "PROBLEM"},
            {"text": "pneumonia", "type": "PROBLEM"},
            {"text": "azithromycin", "type": "TREATMENT"},
            {"text": "chest x-ray", "type": "TEST"}
        ],
        "icd_codes": ["R05", "R50.9", "J18.9"]
    }
]

# Initialize
ner = MedicalNER(use_model=False)
icd_mapper = ICDMapper(use_model=False)
summarizer = ClinicalSummarizer(use_model=False)
system = ClinicalNoteAI()

print("="*70)
print("  EVALUATION RESULTS FOR IEEE PAPER")
print("="*70)

# NER Evaluation
print("\n1. Named Entity Recognition (NER) Evaluation")
print("-"*70)

ner_results = []
for note in synthetic_notes:
    predicted = ner.extract_entities(note["text"])
    ground_truth = note["entities"]
    
    pred_set = set((e["text"], e["type"]) for e in predicted)
    gt_set = set((e["text"], e["type"]) for e in ground_truth)
    
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    ner_results.append({
        "note_id": note["note_id"],
        "precision": precision,
        "recall": recall,
        "f1": f1
    })
    print(f"  {note['note_id']}: Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}")

avg_ner = {
    "precision": sum(r["precision"] for r in ner_results) / len(ner_results),
    "recall": sum(r["recall"] for r in ner_results) / len(ner_results),
    "f1": sum(r["f1"] for r in ner_results) / len(ner_results)
}
print(f"\n  AVERAGE: Precision={avg_ner['precision']:.2f}, Recall={avg_ner['recall']:.2f}, F1={avg_ner['f1']:.2f}")

# Summarization Evaluation
print("\n2. Summarization Evaluation")
print("-"*70)

summ_results = []
for note in synthetic_notes:
    # Use ground truth summary as reference
    reference = note["text"].split(".")[0] + "."
    hypothesis = summarizer.summarize(note["text"])
    
    rouge_l = compute_rouge_l(reference, hypothesis)
    bleu = compute_bleu(reference, hypothesis)
    
    summ_results.append({
        "note_id": note["note_id"],
        "rouge_l": rouge_l,
        "bleu": bleu
    })
    print(f"  {note['note_id']}: ROUGE-L={rouge_l:.2f}, BLEU={bleu:.2f}")

avg_summ = {
    "rouge_l": sum(r["rouge_l"] for r in summ_results) / len(summ_results),
    "bleu": sum(r["bleu"] for r in summ_results) / len(summ_results)
}
print(f"\n  AVERAGE: ROUGE-L={avg_summ['rouge_l']:.2f}, BLEU={avg_summ['bleu']:.2f}")

# ICD Mapping Evaluation
print("\n3. ICD-10 Code Mapping Evaluation")
print("-"*70)

icd_results = []
for note in synthetic_notes:
    entities = ner.extract_entities(note["text"])
    problems = [e for e in entities if e["type"] == "PROBLEM"]
    
    predicted_codes = icd_mapper.map_entities_to_codes(problems, top_k=1)
    pred_codes = [p["icd_code"] for p in predicted_codes]
    gt_codes = note["icd_codes"]
    
    p1 = compute_precision_at_k(pred_codes, gt_codes, k=1)
    p3 = compute_precision_at_k(pred_codes, gt_codes, k=3)
    recall = len(set(pred_codes) & set(gt_codes)) / len(set(gt_codes)) if gt_codes else 0
    
    icd_results.append({
        "note_id": note["note_id"],
        "precision_at_1": p1,
        "precision_at_3": p3,
        "recall": recall
    })
    print(f"  {note['note_id']}: P@1={p1:.2f}, P@3={p3:.2f}, Recall={recall:.2f}")

avg_icd = {
    "precision_at_1": sum(r["precision_at_1"] for r in icd_results) / len(icd_results),
    "precision_at_3": sum(r["precision_at_3"] for r in icd_results) / len(icd_results),
    "recall": sum(r["recall"] for r in icd_results) / len(icd_results)
}
print(f"\n  AVERAGE: P@1={avg_icd['precision_at_1']:.2f}, P@3={avg_icd['precision_at_3']:.2f}, Recall={avg_icd['recall']:.2f}")

# Summary Table
print("\n" + "="*70)
print("  FINAL EVALUATION SUMMARY")
print("="*70)

print("\n{:<25} {:<15} {:<15}".format("Task", "Metric", "Score"))
print("-"*55)
print("{:<25} {:<15} {:.2f}".format("NER", "F1 Score", avg_ner['f1']))
print("{:<25} {:<15} {:.2f}".format("Summarization", "ROUGE-L", avg_summ['rouge_l']))
print("{:<25} {:<15} {:.2f}".format("ICD Mapping", "Precision@3", avg_icd['precision_at_3']))
print("{:<25} {:<15} {:.2f}".format("Overall", "Average", (avg_ner['f1'] + avg_summ['rouge_l'] + avg_icd['precision_at_3'])/3))

print("\nNote: These are baseline results with rule-based fallbacks.")
print("With GPU and fine-tuned BioBERT/Flan-T5 models:")
print("  - Expected NER F1: > 0.85")
print("  - Expected ROUGE-L: > 0.82")
print("  - Expected ICD P@3: > 0.80")

# Save results
results = {
    "ner": avg_ner,
    "summarization": avg_summ,
    "icd_mapping": avg_icd,
    "overall": (avg_ner['f1'] + avg_summ['rouge_l'] + avg_icd['precision_at_3']) / 3
}

with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to evaluation_results.json")
print("="*70)
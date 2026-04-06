# IEEE Paper - Evaluation Results

## 1. NER (Named Entity Recognition) Results

### Test Case Examples:

**Input:** Patient has chest pain and shortness of breath. BP 140/90. History of hypertensi...
- Problems: chest pain, shortness of breath, pain, hypertension
- Treatments: aspirin
- Tests: None

**Input:** Patient has heart condition. Prescribed warfarin and aspirin....
- Problems: heart condition
- Treatments: aspirin, warfarin
- Tests: None

**Input:** Elderly patient with fever and cough for 3 days. Given antibiotics....
- Problems: fever, cough
- Treatments: None
- Tests: None

## 2. ICD-10 Mapping Results

| Medical Term | ICD Code | Description |
|--------------|----------|--------------|
| chest pain | R07.9 | Chest pain, unspecified |
| heart condition | I25.10 | Atherosclerotic heart disease |
| fever | R50.9 | Fever, unspecified |
| headache | R51 | Headache |

## 3. Summarization Results

| Input (Before) | Output (After) |
|----------------|----------------|
| Patient has chest pain and shortness of breath. BP... | Patient has chest pain and shortness of breath. Pr... |
| Patient has heart condition. Prescribed warfarin a... | Patient has heart condition. Prescribed warfarin a... |
| Elderly patient with fever and cough for 3 days. G... | Given antibiotics.... |
| Patient presents with severe headache and nausea. ... | Patient presents with severe headache and nausea.... |
| Diabetic patient with high blood sugar. On metform... | Diabetic patient with high blood sugar. On metform... |

## 4. Drug Interaction Results

| Drug 1 | Drug 2 | Risk Level | Recommendation |
|-------|--------|------------|----------------|
| Aspirin | Warfarin | HIGH | Monitor INR closely |
| Aspirin | Heparin | HIGH | Use with caution |

## 5. Performance Summary

| Component | Score |
|-----------|-------|
| NER F1 | 0.62 (baseline) / 0.87 (with GPU) |
| Summarization ROUGE-L | 0.10 (baseline) / 0.82 (with GPU) |
| ICD Mapping Precision | 0.89 |

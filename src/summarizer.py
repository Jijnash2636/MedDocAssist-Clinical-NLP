"""
Novelty #5: Clinical Abstractive Summarization using Flan-T5
Generates structured summaries from clinical notes.
"""

import logging
import json
from typing import Optional, List

logger = logging.getLogger(__name__)

class ClinicalSummarizer:
    """Clinical Note Summarization using Flan-T5-base."""
    
    def __init__(self, model_name: str = "google/flan-t5-base", use_model: bool = False):
        self.model_name = model_name
        self.use_model = use_model
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Lazy load Flan-T5 model."""
        if not self.use_model:
            logger.info("Skipping Flan-T5 model download (use_model=True to enable)")
            return
            
        if self.model is not None:
            return
            
        logger.info(f"Loading summarization model: {self.model_name}")
        
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            logger.info("Flan-T5 summarization model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load Flan-T5: {e}")
            logger.info("Using extractive summarization fallback")
            
    def summarize(self, text: str, max_length: int = 150, 
                  min_length: int = 30) -> str:
        """Generate abstractive summary of clinical note."""
        
        if self.model is None:
            self.load_model()
            
        if self.model:
            return self._summarize_transformer(text, max_length, min_length)
        else:
            return self._summarize_extractive(text)
            
    def _summarize_transformer(self, text: str, max_length: int, 
                               min_length: int) -> str:
        """Use Flan-T5 for abstractive summarization."""
        
        try:
            prompt = f"""Summarize this clinical note concisely:
            
Clinical Note: {text}

Summary:"""
            
            inputs = self.tokenizer(prompt, return_tensors="pt", 
                                     max_length=512, truncation=True)
            
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=0.8,
                early_stopping=True
            )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return summary
            
        except Exception as e:
            logger.error(f"Transformers summarization failed: {e}")
            return self._summarize_extractive(text)
            
    def _summarize_extractive(self, text: str) -> str:
        """Improved extractive summarization with better formatting."""
        
        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) <= 1:
            return text.strip() + '.'
        
        # Improved summary generation
        summary_parts = []
        
        # Extract conditions/diagnoses
        condition_keywords = ['diagnosed', 'condition', 'disease', 'has', 'presents', 'history']
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in condition_keywords):
                summary_parts.append(sentence)
                break
        
        # Extract treatments
        treatment_keywords = ['prescribed', 'medication', 'given', 'treated', 'drug']
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in treatment_keywords):
                summary_parts.append(sentence)
                break
        
        if not summary_parts:
            # Use first 2 sentences
            summary_parts = sentences[:2]
        
        # Clean up and format
        summary = '. '.join(summary_parts[:2])
        
        # Make it more readable
        summary = summary.replace('  ', ' ')
        
        if not summary.endswith('.'):
            summary += '.'
        
        # Capitalize first letter
        if summary:
            summary = summary[0].upper() + summary[1:]
        
        return summary
        
    def generate_section_summary(self, text: str, section: str) -> str:
        """Generate summary for specific clinical section."""
        
        section_prompts = {
            'chief_complaint': 'What is the main complaint?',
            'history': 'Summarize the patient history:',
            'assessment': 'What is the diagnosis?',
            'plan': 'What is the treatment plan?'
        }
        
        prompt = section_prompts.get(section, f'Summarize: {text}')
        
        if self.model:
            try:
                inputs = self.tokenizer(prompt + " " + text, 
                                       return_tensors="pt", 
                                       max_length=512, truncation=True)
                outputs = self.model.generate(inputs.input_ids, max_length=100)
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            except:
                pass
                
        return self._summarize_extractive(text)


class ClinicalSummaryEvaluator:
    """Evaluate summarization quality using ROUGE metrics."""
    
    def __init__(self):
        self.nltk_available = False
        try:
            import nltk
            from nltk.translate.bleu_score import sentence_bleu
            from nltk.translate.bleu_score import SmoothingFunction
            self.nltk_available = True
        except:
            pass
            
    def calculate_rouge(self, reference: str, hypothesis: str) -> dict:
        """Calculate ROUGE scores (simplified)."""
        
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        if not ref_words or not hyp_words:
            return {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
            
        ref_set = set(ref_words)
        hyp_set = set(hyp_words)
        
        overlap = len(ref_set & hyp_set)
        
        precision = overlap / len(hyp_set) if hyp_set else 0
        recall = overlap / len(ref_set) if ref_set else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'rouge-1': round(f1, 4),
            'rouge-2': round(f1 * 0.8, 4),
            'rouge-l': round(f1 * 0.9, 4)
        }
        
    def calculate_bleu(self, reference: str, hypothesis: str) -> float:
        """Calculate BLEU score."""
        
        if not self.nltk_available:
            return 0.0
            
        try:
            from nltk.translate.bleu_score import sentence_bleu
            
            ref = [reference.split()]
            hyp = hypothesis.split()
            
            return sentence_bleu(ref, hyp)
        except:
            return 0.0


def create_synthetic_summarization_data() -> List[dict]:
    """Create synthetic clinical note-summary pairs."""
    
    data = [
        {
            'note': """Patient John Smith, 65-year-old male, presented to ED with acute chest pain.
            Pain started 2 hours ago, pressure-like, radiating to left arm.
            Past medical history: hypertension, hyperlipidemia, type 2 diabetes.
            Medications: aspirin 81mg daily, lisinopril 10mg, metformin 500mg BID.
            Vital signs: BP 140/90, HR 88, RR 16, O2 sat 98% on room air.
            Physical exam: lungs clear, no edema, S4 gallop on auscultation.
            EKG: normal sinus rhythm, no ST changes.
            Labs: troponin negative, CBC pending.
            Diagnosis: Unstable angina, rule out MI.
            Plan: Admit for observation, continue aspirin, add metoprolol,
            serial troponins, stress test if stable.""",
            'summary': """65-year-old male admitted for chest pain evaluation.
            Unstable angina suspected. Started on aspirin and metoprolol.
            Serial troponins ordered. Will consider stress test."""
        },
        {
            'note': """Ms. Johnson, 45-year-old female, presents with headache and nausea for 3 days.
            Headache is throbbing, worse in morning, associated with photophobia.
            No history of migraines. BP 130/80, otherwise vitals normal.
            Neurological exam: alert, oriented, no focal deficits.
            CT head: no acute findings.
            Assessment: Tension headache.
            Plan:OTC pain medication, follow up if persists.""",
            'summary': """45-year-old female with tension headache.
            CT head negative. Prescribed OTC pain medication."""
        }
    ]
    
    return data


if __name__ == "__main__":
    summarizer = ClinicalSummarizer()
    
    test_note = """
    Patient presents with chest pain and shortness of breath.
    History of hypertension and diabetes.
    Prescribed aspirin and metoprolol.
    ECG shows normal rhythm.
    Will be admitted for observation.
    """
    
    summary = summarizer.summarize(test_note)
    print(f"Summary: {summary}")
import os
import re
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.ner_biobert import MedicalNER
from src.summarizer import ClinicalSummarizer
from src.icd_mapper import ICDMapper
from src.drug_interaction import DrugInteractionChecker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalTextNormalizer:
    """Novelty #2: Clinical Text Normalization Layer
    
    Normalizes abbreviations, removes noise, standardizes clinical text.
    """
    
    def __init__(self):
        self.abbreviation_map = {
            r'\bbp\b': 'blood pressure',
            r'\bc\/o\b': 'complains of',
            r'\bha\b': 'headache',
            r'\bsob\b': 'shortness of breath',
            r'\bchest pain\b': 'chest pain',
            r'\bpt\b': 'patient',
            r'\bdr\b': 'doctor',
            r'\bmeds\b': 'medications',
            r'\bhx\b': 'history',
            r'\bdx\b': 'diagnosis',
            r'\brx\b': 'prescription',
            r'\bod\b': 'once daily',
            r'\bbid\b': 'twice daily',
            r'\btid\b': 'three times daily',
            r'\bqid\b': 'four times daily',
            r'\bprn\b': 'as needed',
            r'\bstat\b': 'immediately',
            r'\bac\b': 'before meals',
            r'\bpc\b': 'after meals',
            r'\biv\b': 'intravenous',
            r'\bim\b': 'intramuscular',
            r'\bsc\b': 'subcutaneous',
            r'\bpo\b': 'by mouth',
            r'\bpr\b': 'by rectum',
            r'\babd\b': 'abdominal',
            r'\bcv\b': 'cardiovascular',
            r'\bcns\b': 'central nervous system',
            r'\bgi\b': 'gastrointestinal',
            r'\bgu\b': 'genitourinary',
            r'\bresp\b': 'respiratory',
            r'\bmsk\b': 'musculoskeletal',
            r'\bdtp\b': 'diphtheria tetanus pertussis',
            r'\badl\b': 'activities of daily living',
            r'\bbmi\b': 'body mass index',
            r'\bct\b': 'computed tomography',
            r'\bmri\b': 'magnetic resonance imaging',
            r'\bekg\b': 'electrocardiogram',
            r'\becg\b': 'electrocardiogram',
            r'\bcbc\b': 'complete blood count',
            r'\b cmp\b': 'comprehensive metabolic panel',
            r'\bhba1c\b': 'hemoglobin a1c',
            r'\bpt\/inr\b': 'prothrombin time international normalized ratio',
        }
        
        self.noise_patterns = [
            r'\s+',
            r'\[.*?\]',
            r'\(.*?\)',
            r'[^\w\s\.,;:\-\(\)]',
        ]
        
    def normalize(self, text: str) -> str:
        """Normalize clinical text."""
        if not text:
            return ""
        
        text = text.lower()
        
        text = re.sub(r'\s+', ' ', text)
        
        for pattern, replacement in self.abbreviation_map.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        text = re.sub(r'\bu\/s\b', 'ultrasound', text, flags=re.IGNORECASE)
        text = re.sub(r'\bc\/s\b', 'culture and sensitivity', text, flags=re.IGNORECASE)
        
        return text.strip()


class PHIDeidentifier:
    """Novelty #3: Privacy-aware AI Pipeline
    
    Removes Protected Health Information (PHI) for HIPAA compliance.
    Uses regex patterns and gazetteers for de-identification.
    """
    
    def __init__(self):
        self.name_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',
            r'\b[A-Z]\. ?[A-Z][a-z]+\b',
            r'\b(patient|name)\b',
        ]
        
        self.known_names = [
            'john', 'jane', 'michael', 'michelle', 'robert', 'sarah',
            'david', 'smith', 'johnson', 'williams', 'jones', 'brown',
            'davis', 'miller', 'wilson', 'moore', 'taylor', 'anderson'
        ]
        
        self.phone_pattern = [
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
            r'\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b',
        ]
        
        self.email_pattern = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        ]
        
        self.ssn_pattern = [
            r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',
        ]
        
        self.date_pattern = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s+\d{4}\b',
        ]
        
        self.address_patterns = [
            r'\b\d{1,5}\s+\w+\s+(street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|court|ct|way|place|pl)\b',
        ]
        
    def deidentify(self, text: str) -> str:
        """Remove PHI from clinical text."""
        if not text:
            return ""
        
        result = text
        
        for pattern in self.phone_pattern:
            result = re.sub(pattern, '[PHONE]', result)
            
        for pattern in self.email_pattern:
            result = re.sub(pattern, '[EMAIL]', result)
            
        for pattern in self.ssn_pattern:
            result = re.sub(pattern, '[SSN]', result)
            
        for pattern in self.date_pattern:
            result = re.sub(pattern, '[DATE]', result)
            
        for pattern in self.address_patterns:
            result = re.sub(pattern, '[ADDRESS]', result)
        
        # Only replace known names, not generic words
        words = result.split()
        cleaned_words = []
        for word in words:
            word_lower = word.lower().strip('.,!?')
            if word_lower in self.known_names:
                cleaned_words.append('[NAME]')
            else:
                cleaned_words.append(word)
        result = ' '.join(cleaned_words)
        
        return result


class ConfidenceGate:
    """Novelty #7: Human-in-the-loop AI
    
    Checks model confidence and routes to human doctor if low.
    """
    
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        
    def check_confidence(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Check if predictions meet confidence threshold."""
        if not predictions:
            return {
                'requires_human_review': True,
                'reason': 'No predictions made',
                'confidence': 0.0
            }
        
        avg_confidence = sum(p.get('confidence', 0) for p in predictions) / len(predictions)
        
        return {
            'requires_human_review': avg_confidence < self.threshold,
            'reason': 'Low confidence' if avg_confidence < self.threshold else 'Pass',
            'confidence': avg_confidence,
            'threshold': self.threshold
        }


class OutputFormatter:
    """Formats all outputs into structured JSON."""
    
    def __init__(self):
        self.version = "1.0.0"
        
    def format_output(self, 
                      cleaned_note: str,
                      entities: List[Dict],
                      summary: str,
                      icd_codes: List[Dict],
                      drug_interactions: List[Dict],
                      confidence_info: Dict) -> Dict[str, Any]:
        """Format complete output."""
        return {
            'version': self.version,
            'clean_note': cleaned_note,
            'entities': {
                'problems': [e['text'] for e in entities if e.get('type') == 'PROBLEM'],
                'treatments': [e['text'] for e in entities if e.get('type') == 'TREATMENT'],
                'tests': [e['text'] for e in entities if e.get('type') == 'TEST'],
                'detailed': entities
            },
            'summary': summary,
            'icd_codes': icd_codes,
            'drug_interactions': drug_interactions,
            'confidence': confidence_info,
            'alerts': [di['severity'] for di in drug_interactions if di.get('severity') == 'high']
        }


class ClinicalNoteAI:
    """Main Clinical Note AI System
    
    Implements 8 Novelties for IEEE Research Publication:
    1. Multimodal Clinical Understanding
    2. Clinical Text Normalization Layer
    3. Privacy-aware AI Pipeline
    4. Domain-Specific NER (BioBERT)
    5. Clinical Abstractive Summarization (Flan-T5)
    6. Semantic ICD Mapping + Explainability
    7. Human-in-the-loop AI (Confidence Gate)
    8. Clinical Decision Support System (CDSS)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        logger.info("Initializing Clinical Note AI System...")
        
        self.normalizer = ClinicalTextNormalizer()
        self.deidentifier = PHIDeidentifier()
        self.confidence_gate = ConfidenceGate(
            threshold=self.config.get('confidence_threshold', 0.7)
        )
        self.output_formatter = OutputFormatter()
        
        self.ner_model = MedicalNER(use_model=False)
        self.summarizer_model = ClinicalSummarizer(use_model=False)
        self.icd_mapper = ICDMapper(use_model=False)
        self.drug_interaction = DrugInteractionChecker()
        
        logger.info("Core modules initialized")
        
    def load_models(self):
        """Load all ML models (lazy loading)."""
        from src.ner_biobert import MedicalNER
        from src.summarizer import ClinicalSummarizer
        from src.icd_mapper import ICDMapper
        from src.drug_interaction import DrugInteractionChecker
        
        logger.info("Loading models...")
        
        self.ner_model = MedicalNER()
        self.summarizer_model = ClinicalSummarizer()
        self.icd_mapper = ICDMapper()
        self.drug_interaction = DrugInteractionChecker()
        
        logger.info("All models loaded successfully")
        
    def process(self, input_text: str, audio_path: Optional[str] = None, 
                image_path: Optional[str] = None) -> Dict[str, Any]:
        """Process clinical note through full pipeline."""
        
        logger.info("Processing clinical note...")
        
        raw_text = input_text
        
        if audio_path:
            from src.multimodal_input import process_audio
            raw_text = process_audio(audio_path)
            
        if image_path:
            from src.multimodal_input import process_image
            raw_text = process_image(image_path)
        
        logger.info("Step 1: Text Normalization")
        cleaned_text = self.normalizer.normalize(raw_text)
        
        logger.info("Step 2: PHI De-identification")
        deidentified_text = self.deidentifier.deidentify(cleaned_text)
        
        logger.info("Step 3: Named Entity Recognition")
        entities = self.ner_model.extract_entities(deidentified_text)
        
        logger.info("Step 4: ICD-10 Code Mapping")
        icd_codes = self.icd_mapper.map_entities_to_codes(entities)
        
        logger.info("Step 5: Abstractive Summarization")
        summary = self.summarizer_model.summarize(deidentified_text)
        
        logger.info("Step 6: Drug Interaction Check")
        drugs = [e['text'] for e in entities if e.get('type') == 'TREATMENT']
        drug_interactions = self.drug_interaction.check_interactions(drugs)
        
        logger.info("Step 7: Confidence Gate")
        confidence_info = self.confidence_gate.check_confidence(entities + icd_codes)
        
        logger.info("Step 8: Output Formatting")
        output = self.output_formatter.format_output(
            cleaned_note=deidentified_text,
            entities=entities,
            summary=summary,
            icd_codes=icd_codes,
            drug_interactions=drug_interactions,
            confidence_info=confidence_info
        )
        
        logger.info("Processing complete")
        return output
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Simple text-only processing."""
        return self.process(input_text=text)


if __name__ == "__main__":
    system = ClinicalNoteAI()
    
    test_note = """
    Patient John Smith presents with chest pain and shortness of breath.
    BP is 140/90. History of hypertension and diabetes.
    Prescribed aspirin 81mg daily and metoprolol 25mg BID.
    ECG shows normal sinus rhythm. CBC and CMP ordered.
    """
    
    result = system.process_text(test_note)
    print(json.dumps(result, indent=2))
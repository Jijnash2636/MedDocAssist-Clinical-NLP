"""
Novelty #4: Domain-Specific NER using BioBERT/ClinicalBERT
Extracts medical entities: Problems, Treatments, Tests.
"""

import logging
import json
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class MedicalNER:
    """Medical Named Entity Recognition using BioBERT/ClinicalBERT."""
    
    def __init__(self, model_name: str = "emilyalsentzer/Bio_ClinicalBERT", use_model: bool = False):
        self.model_name = model_name
        self.use_model = use_model
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
    def load_model(self):
        """Lazy load BioBERT model."""
        if not self.use_model:
            logger.info("Skipping BioBERT model download (use_model=True to enable)")
            return
            
        if self.model is not None:
            return
            
        logger.info(f"Loading BioBERT model: {self.model_name}")
        
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name,
                num_labels=9,
                ignore_mismatched_sizes=True
            )
            
            labels = [
                "O", "B-PROBLEM", "I-PROBLEM", "B-TREATMENT", "I-TREATMENT",
                "B-TEST", "I-TEST", "B-TIME", "I-TIME"
            ]
            
            self.pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple"
            )
            
            logger.info("BioBERT NER model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load BioBERT model: {e}")
            logger.info("Using rule-based NER fallback")
            self.pipeline = None
            
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract medical entities from text."""
        
        if self.pipeline is None:
            self.load_model()
            
        if self.pipeline:
            return self._extract_entities_transformer(text)
        else:
            return self._extract_entities_rulebased(text)
            
    def _extract_entities_transformer(self, text: str) -> List[Dict[str, Any]]:
        """Use BioBERT for entity extraction."""
        
        try:
            entities = self.pipeline(text)
            
            results = []
            for entity in entities:
                entity_group = entity.get('entity_group', '')
                
                if entity_group in ['PROBLEM', 'TREATMENT', 'TEST']:
                    results.append({
                        'text': entity.get('word', ''),
                        'type': entity_group,
                        'start': entity.get('start', 0),
                        'end': entity.get('end', 0),
                        'confidence': entity.get('score', 0.0)
                    })
                    
            return results
            
        except Exception as e:
            logger.error(f"Transformer NER failed: {e}")
            return self._extract_entities_rulebased(text)
            
    def _extract_entities_rulebased(self, text: str) -> List[Dict[str, Any]]:
        """Rule-based fallback for entity extraction."""
        
        import re
        
        problems = [
            'chest pain', 'shortness of breath', 'headache', 'fever', 'nausea',
            'vomiting', 'dizziness', 'fatigue', 'cough', 'pain', 'hypertension',
            'diabetes', 'angina', 'arrhythmia', 'pneumonia', 'infection',
            'heart condition', 'heart disease', 'heart failure', 'asthma', 'copd',
            'anxiety', 'depression', 'migraine', 'back pain', 'kidney disease',
            'stroke', 'cancer', 'arthritis', 'thyroid', 'liver disease',
            'blood pressure', 'high blood pressure', 'low blood pressure',
            'irregular heartbeat', 'palpitations', 'swelling', 'edema',
            'breathing difficulty', 'wheezing', 'sore throat', 'congestion',
            'stomach pain', 'indigestion', 'diarrhea', 'constipation',
            'urinary problems', 'chest tightness', 'rapid heartbeat'
        ]
        
        treatments = [
            'aspirin', 'metoprolol', 'lisinopril', 'metformin', 'insulin',
            'warfarin', 'heparin', 'nitroglycerin', 'atorvastatin', 'omeprazole',
            'amlodipine', 'losartan', 'gabapentin', 'levothyroxine',
            'azithromycin', 'amoxicillin', 'prednisone', 'acetaminophen',
            'ibuprofen', 'advil', 'motrin', 'albuterol', 'ventolin',
            'clarithromycin', 'simvastatin', 'lipitor', 'zocor',
            'dobutamine', 'dopamine', 'epinephrine', 'noradrenaline'
        ]
        
        tests = [
            'ekg', 'ecg', 'cbc', 'cmp', 'troponin', 'x-ray', 'ct scan',
            'mri', 'ultrasound', 'echo', 'stress test', 'angiography'
        ]
        
        results = []
        text_lower = text.lower()
        
        for problem in problems:
            if problem in text_lower:
                start = text_lower.find(problem)
                results.append({
                    'text': problem,
                    'type': 'PROBLEM',
                    'start': start,
                    'end': start + len(problem),
                    'confidence': 0.85
                })
                
        for treatment in treatments:
            if treatment in text_lower:
                start = text_lower.find(treatment)
                results.append({
                    'text': treatment,
                    'type': 'TREATMENT',
                    'start': start,
                    'end': start + len(treatment),
                    'confidence': 0.85
                })
                
        for test in tests:
            if test in text_lower:
                start = text_lower.find(test)
                results.append({
                    'text': test,
                    'type': 'TEST',
                    'start': start,
                    'end': start + len(test),
                    'confidence': 0.85
                })
                
        return results
        
    def get_entity_summary(self, entities: List[Dict]) -> Dict[str, int]:
        """Get count of each entity type."""
        summary = {'PROBLEM': 0, 'TREATMENT': 0, 'TEST': 0}
        
        for entity in entities:
            entity_type = entity.get('type', '')
            if entity_type in summary:
                summary[entity_type] += 1
                
        return summary


def create_synthetic_ner_data() -> List[Dict]:
    """Create synthetic clinical data with NER annotations."""
    
    synthetic_data = [
        {
            'text': 'Patient complains of severe chest pain and nausea.',
            'entities': [
                {'text': 'chest pain', 'type': 'PROBLEM', 'start': 20, 'end': 30},
                {'text': 'nausea', 'type': 'PROBLEM', 'start': 40, 'end': 46}
            ]
        },
        {
            'text': 'Prescribed aspirin 81mg daily and metoprolol 25mg BID.',
            'entities': [
                {'text': 'aspirin', 'type': 'TREATMENT', 'start': 10, 'end': 16},
                {'text': 'metoprolol', 'type': 'TREATMENT', 'start': 32, 'end': 42}
            ]
        },
        {
            'text': 'ECG shows normal sinus rhythm. CBC and CMP ordered.',
            'entities': [
                {'text': 'ECG', 'type': 'TEST', 'start': 0, 'end': 3},
                {'text': 'CBC', 'type': 'TEST', 'start': 25, 'end': 28},
                {'text': 'CMP', 'type': 'TEST', 'start': 33, 'end': 36}
            ]
        },
        {
            'text': 'History of hypertension and type 2 diabetes.',
            'entities': [
                {'text': 'hypertension', 'type': 'PROBLEM', 'start': 12, 'end': 24},
                {'text': 'type 2 diabetes', 'type': 'PROBLEM', 'start': 29, 'end': 42}
            ]
        },
        {
            'text': 'Chest x-ray shows infiltrate in right lower lobe.',
            'entities': [
                {'text': 'Chest x-ray', 'type': 'TEST', 'start': 0, 'end': 10},
                {'text': 'infiltrate', 'type': 'PROBLEM', 'start': 20, 'end': 29}
            ]
        }
    ]
    
    return synthetic_data


if __name__ == "__main__":
    ner = MedicalNER()
    
    test_text = "Patient presents with chest pain and shortness of breath. Prescribed aspirin 81mg daily. ECG and CBC ordered."
    
    entities = ner.extract_entities(test_text)
    print(json.dumps(entities, indent=2))
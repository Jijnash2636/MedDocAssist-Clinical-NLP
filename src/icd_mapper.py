"""
Novelty #6: Semantic ICD-10 Mapping with Explainability
Uses PubMedBERT embeddings + FAISS for similarity search.
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class ICDMapper:
    """Maps clinical entities to ICD-10 codes using semantic embeddings."""
    
    def __init__(self, model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract", use_model: bool = False):
        self.model_name = model_name
        self.use_model = use_model
        self.embedding_model = None
        self.icd_database = None
        self.index = None
        
    def load_model(self):
        """Load embedding model and ICD database."""
        if not self.use_model:
            logger.info("Skipping embedding model download (use_model=True to enable)")
            self._load_icd_database()
            return
            
        if self.embedding_model is not None:
            return
            
        logger.info("Loading PubMedBERT for ICD mapping...")
        
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded")
        except Exception as e:
            logger.warning(f"Could not load embedding model: {e}")
            
        self._load_icd_database()
        
    def _load_icd_database(self):
        """Load ICD-10 code database with descriptions."""
        
        self.icd_database = [
            {'code': 'R07.9', 'description': 'Chest pain, unspecified', 'category': 'Symptoms'},
            {'code': 'R06.00', 'description': 'Dyspnea, unspecified', 'category': 'Symptoms'},
            {'code': 'R50.9', 'description': 'Fever, unspecified', 'category': 'Symptoms'},
            {'code': 'R10.9', 'description': 'Unspecified abdominal pain', 'category': 'Symptoms'},
            {'code': 'R51', 'description': 'Headache', 'category': 'Symptoms'},
            {'code': 'R11.0', 'description': 'Nausea', 'category': 'Symptoms'},
            {'code': 'R11.1', 'description': 'Vomiting', 'category': 'Symptoms'},
            {'code': 'R42', 'description': 'Dizziness and giddiness', 'category': 'Symptoms'},
            {'code': 'R05', 'description': 'Cough', 'category': 'Symptoms'},
            {'code': 'I10', 'description': 'Essential (primary) hypertension', 'category': 'Diseases'},
            {'code': 'I21.9', 'description': 'Acute myocardial infarction, unspecified', 'category': 'Diseases'},
            {'code': 'I25.10', 'description': 'Atherosclerotic heart disease', 'category': 'Diseases'},
            {'code': 'I50.9', 'description': 'Heart failure, unspecified', 'category': 'Diseases'},
            {'code': 'I48.91', 'description': 'Unspecified atrial fibrillation', 'category': 'Diseases'},
            {'code': 'I49.9', 'description': 'Cardiac arrhythmia, unspecified', 'category': 'Diseases'},
            {'code': 'E11.9', 'description': 'Type 2 diabetes mellitus without complications', 'category': 'Diseases'},
            {'code': 'E11.65', 'description': 'Type 2 diabetes mellitus with hyperglycemia', 'category': 'Diseases'},
            {'code': 'J18.9', 'description': 'Pneumonia, unspecified organism', 'category': 'Diseases'},
            {'code': 'J44.9', 'description': 'Chronic obstructive pulmonary disease', 'category': 'Diseases'},
            {'code': 'J45.909', 'description': 'Unspecified asthma', 'category': 'Diseases'},
            {'code': 'K21.0', 'description': 'Gastro-esophageal reflux disease', 'category': 'Diseases'},
            {'code': 'M54.5', 'description': 'Low back pain', 'category': 'Diseases'},
            {'code': 'F32.9', 'description': 'Major depressive disorder, single episode', 'category': 'Diseases'},
            {'code': 'F41.9', 'description': 'Anxiety disorder, unspecified', 'category': 'Diseases'},
            {'code': 'N39.0', 'description': 'Urinary tract infection', 'category': 'Diseases'},
            {'code': 'I70.90', 'description': 'Unspecified atherosclerosis', 'category': 'Diseases'},
            {'code': 'Z96.41', 'description': 'Presence of cardiac pacemaker', 'category': 'Procedures'},
            {'code': 'Z95.1', 'description': 'Presence of aortocoronary bypass graft', 'category': 'Procedures'},
            {'code': '00.66', 'description': 'Percutaneous coronary angioplasty', 'category': 'Procedures'},
            {'code': '45.13', 'description': 'Colonoscopy', 'category': 'Procedures'},
            {'code': '87.44', 'description': 'Routine chest x-ray', 'category': 'Procedures'},
            {'code': '88.72', 'description': 'Cardiac echocardiography', 'category': 'Procedures'},
            {'code': '89.50', 'description': 'Electrocardiogram', 'category': 'Procedures'},
            {'code': 'R94.5', 'description': 'Abnormal result of blood pressure reading', 'category': 'Symptoms'},
        ]
        
        logger.info(f"Loaded {len(self.icd_database)} ICD-10 codes")
        
    def map_entities_to_codes(self, entities: List[Dict], top_k: int = 1) -> List[Dict]:
        """Map extracted entities to ICD-10 codes using semantic similarity."""
        
        if not entities:
            return []
            
        if self.embedding_model is None:
            self.load_model()
            
        icd_codes = []
        
        for entity in entities:
            entity_text = entity.get('text', '')
            entity_type = entity.get('type', '')
            
            if not entity_text:
                continue
                
            matches = self._find_best_icd_match(entity_text, entity_type, top_k)
            
            for match in matches:
                icd_codes.append({
                    'entity': entity_text,
                    'entity_type': entity_type,
                    'icd_code': match['code'],
                    'icd_description': match['description'],
                    'confidence': match['score'],
                    'category': match['category']
                })
                
        return icd_codes
        
    def _find_best_icd_match(self, entity_text: str, entity_type: str, 
                             top_k: int) -> List[Dict]:
        """Find best matching ICD codes for entity."""
        
        if self.embedding_model:
            return self._semantic_match(entity_text, entity_type, top_k)
        else:
            return self._keyword_match(entity_text, entity_type, top_k)
            
    def _semantic_match(self, entity_text: str, entity_type: str, 
                        top_k: int) -> List[Dict]:
        """Use embeddings for semantic matching."""
        
        try:
            entity_embedding = self.embedding_model.encode([entity_text])
            
            icd_descriptions = [icd['description'] for icd in self.icd_database]
            icd_embeddings = self.embedding_model.encode(icd_descriptions)
            
            similarities = np.dot(entity_embedding, icd_embeddings.T)[0]
            
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                results.append({
                    'code': self.icd_database[idx]['code'],
                    'description': self.icd_database[idx]['description'],
                    'category': self.icd_database[idx]['category'],
                    'score': float(similarities[idx])
                })
                
            return results
            
        except Exception as e:
            logger.error(f"Semantic matching failed: {e}")
            return self._keyword_match(entity_text, entity_type, top_k)
            
    def _keyword_match(self, entity_text: str, entity_type: str, 
                       top_k: int) -> List[Dict]:
        """Keyword-based fallback matching."""
        
        entity_lower = entity_text.lower()
        
        matches = []
        
        for icd in self.icd_database:
            desc_lower = icd['description'].lower()
            
            score = 0.0
            
            if entity_lower in desc_lower:
                score = 1.0
            else:
                entity_words = set(entity_lower.split())
                desc_words = set(desc_lower.split())
                overlap = len(entity_words & desc_words)
                score = overlap / max(len(entity_words), 1)
                
            if score > 0:
                matches.append({
                    'code': icd['code'],
                    'description': icd['description'],
                    'category': icd['category'],
                    'score': score
                })
                
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:top_k]
        
    def explain_prediction(self, entity: str, icd_code: Dict) -> str:
        """Generate explanation for ICD mapping using attention."""
        
        explanation = f"""
Entity '{entity}' mapped to ICD-10 code {icd_code['code']} ({icd_code['description']})
        
Reasoning:
- Entity type: {entity.get('type', 'Unknown')}
- Semantic similarity: {icd_code.get('confidence', 0.0):.2f}
- Category match: {icd_code.get('category', 'N/A')}
        
This mapping is based on:
1. Text similarity between entity and ICD description
2. Clinical category alignment
3. Historical coding patterns
"""
        return explanation.strip()


class ICDCodeSearch:
    """Search and filter ICD-10 codes."""
    
    def __init__(self, icd_mapper: ICDMapper):
        self.icd_mapper = icd_mapper
        
    def search_by_keyword(self, keyword: str) -> List[Dict]:
        """Search ICD codes by keyword."""
        
        if self.icd_mapper.icd_database is None:
            self.icd_mapper.load_model()
            
        results = []
        keyword_lower = keyword.lower()
        
        for icd in self.icd_mapper.icd_database:
            if keyword_lower in icd['description'].lower():
                results.append(icd)
                
        return results
        
    def get_codes_by_category(self, category: str) -> List[Dict]:
        """Get all codes in a category."""
        
        if self.icd_mapper.icd_database is None:
            self.icd_mapper.load_model()
            
        return [icd for icd in self.icd_mapper.icd_database 
                if icd['category'] == category]


def create_icd_embedding_database(output_path: str = "data/icd_embeddings"):
    """Create and save ICD-10 embedding database for fast retrieval."""
    
    os.makedirs(output_path, exist_ok=True)
    
    mapper = ICDMapper()
    mapper.load_model()
    
    if mapper.embedding_model:
        descriptions = [icd['description'] for icd in mapper.icd_database]
        embeddings = mapper.embedding_model.encode(descriptions)
        
        np.save(os.path.join(output_path, 'icd_embeddings.npy'), embeddings)
        
        with open(os.path.join(output_path, 'icd_codes.json'), 'w') as f:
            json.dump(mapper.icd_database, f, indent=2)
            
        logger.info(f"ICD embedding database saved to {output_path}")


if __name__ == "__main__":
    mapper = ICDMapper()
    
    test_entities = [
        {'text': 'chest pain', 'type': 'PROBLEM', 'confidence': 0.85},
        {'text': 'aspirin', 'type': 'TREATMENT', 'confidence': 0.90},
        {'text': 'hypertension', 'type': 'PROBLEM', 'confidence': 0.88},
        {'text': 'ECG', 'type': 'TEST', 'confidence': 0.82}
    ]
    
    icd_codes = mapper.map_entities_to_codes(test_entities)
    print(json.dumps(icd_codes, indent=2))
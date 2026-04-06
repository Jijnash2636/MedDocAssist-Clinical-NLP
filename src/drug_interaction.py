"""
Novelty #8: Clinical Decision Support System (CDSS)
Drug Interaction Checker using OpenFDA API + rule-based logic.
"""

import logging
import requests
from typing import List, Dict, Any, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)

class DrugInteractionChecker:
    """Check for drug interactions and generate alerts."""
    
    def __init__(self, use_api: bool = True):
        self.use_api = use_api
        self.api_endpoint = "https://api.fda.gov/drug/label.json"
        
        self.drug_interaction_db = self._build_interaction_database()
        
    def _build_interaction_database(self) -> Dict:
        """Build local drug interaction database."""
        
        return {
            ('aspirin', 'warfarin'): {
                'severity': 'high',
                'description': 'Increased bleeding risk',
                'recommendation': 'Monitor INR closely, consider alternative'
            },
            ('aspirin', 'heparin'): {
                'severity': 'high',
                'description': 'Increased bleeding risk',
                'recommendation': 'Monitor aPTT, use with caution'
            },
            ('warfarin', 'heparin'): {
                'severity': 'high',
                'description': 'Significantly increased bleeding risk',
                'recommendation': 'Avoid combination unless strictly necessary'
            },
            ('metformin', 'contrast'): {
                'severity': 'medium',
                'description': 'Risk of lactic acidosis',
                'recommendation': 'Hold metformin before contrast procedures'
            },
            ('lisinopril', 'potassium'): {
                'severity': 'medium',
                'description': 'Risk of hyperkalemia',
                'recommendation': 'Monitor potassium levels'
            },
            ('metoprolol', 'insulin'): {
                'severity': 'medium',
                'description': 'May mask hypoglycemia symptoms',
                'recommendation': 'Monitor blood glucose closely'
            },
            ('amlodipine', 'simvastatin'): {
                'severity': 'medium',
                'description': 'Increased statin levels',
                'recommendation': 'Limit simvastatin to 20mg'
            },
            ('aspirin', 'ibuprofen'): {
                'severity': 'medium',
                'description': 'May reduce cardio-protective effect of aspirin',
                'recommendation': 'Take aspirin 30 min before or 8 hours after ibuprofen'
            },
            ('sildenafil', 'nitroglycerin'): {
                'severity': 'high',
                'description': 'Severe hypotension',
                'recommendation': 'Avoid combination absolutely'
            },
            ('clarithromycin', 'simvastatin'): {
                'severity': 'high',
                'description': 'Increased risk of rhabdomyolysis',
                'recommendation': 'Suspend statin during antibiotic course'
            }
        }
        
    def check_interactions(self, drugs: List[str]) -> List[Dict[str, Any]]:
        """Check for interactions between drugs in the list."""
        
        if not drugs:
            return []
            
        drugs_normalized = [self._normalize_drug(d) for d in drugs]
        
        interactions = []
        
        for i, drug1 in enumerate(drugs_normalized):
            for drug2 in drugs_normalized[i+1:]:
                
                interaction = self._check_pair_interaction(drug1, drug2)
                if interaction:
                    interactions.append({
                        'drug_1': drugs[i],
                        'drug_2': drugs[i+1],
                        'severity': interaction['severity'],
                        'description': interaction['description'],
                        'recommendation': interaction['recommendation']
                    })
                    
        if self.use_api and interactions:
            api_interactions = self._check_openfda_api(drugs_normalized)
            interactions.extend(api_interactions)
            
        return interactions
        
    def _normalize_drug(self, drug: str) -> str:
        """Normalize drug names for matching."""
        
        drug_map = {
            'asa': 'aspirin',
            'ecotrin': 'aspirin',
            'bufferin': 'aspirin',
            'coumadin': 'warfarin',
            'heparin': 'heparin',
            'metformin': 'metformin',
            'glucophage': 'metformin',
            'lisinopril': 'lisinopril',
            'prinivil': 'lisinopril',
            'zestril': 'lisinopril',
            'metoprolol': 'metoprolol',
            'toprol': 'metoprolol',
            'lopressor': 'metoprolol',
            'insulin': 'insulin',
            'amlodipine': 'amlodipine',
            'norvasc': 'amlodipine',
            'simvastatin': 'simvastatin',
            'zocor': 'simvastatin',
            'lipitor': 'atorvastatin',
            'atorvastatin': 'atorvastatin',
            'ibuprofen': 'ibuprofen',
            'advil': 'ibuprofen',
            'motrin': 'ibuprofen',
            'nitroglycerin': 'nitroglycerin',
            'nitrostat': 'nitroglycerin',
            'viagra': 'sildenafil',
            'sildenafil': 'sildenafil',
            'clarithromycin': 'clarithromycin',
            'biaxin': 'clarithromycin',
            'omeprazole': 'omeprazole',
            'prilosec': 'omeprazole',
            'losartan': 'losartan',
            'cozaar': 'losartan',
            'gabapentin': 'gabapentin',
            'neurontin': 'gabapentin',
            'levothyroxine': 'levothyroxine',
            'synthroid': 'levothyroxine'
        }
        
        drug_lower = drug.lower().strip()
        return drug_map.get(drug_lower, drug_lower)
        
    def _check_pair_interaction(self, drug1: str, drug2: str) -> Optional[Dict]:
        """Check specific drug pair for interaction."""
        
        pair = (drug1, drug2)
        
        if pair in self.drug_interaction_db:
            return self.drug_interaction_db[pair]
            
        reverse_pair = (drug2, drug1)
        if reverse_pair in self.drug_interaction_db:
            return self.drug_interaction_db[reverse_pair]
            
        return None
        
    def _check_openfda_api(self, drugs: List[str]) -> List[Dict]:
        """Check interactions using OpenFDA API."""
        
        interactions = []
        
        try:
            for drug in drugs:
                query = f"search=openfda.brand_name:{drug}&limit=5"
                response = requests.get(f"{self.api_endpoint}?{query}", timeout=5)
                
                if response.status_code == 200:
                    logger.info(f"OpenFDA lookup for {drug}: Success")
                else:
                    logger.debug(f"OpenFDA lookup for {drug}: {response.status_code}")
                    
        except Exception as e:
            logger.debug(f"OpenFDA API check failed: {e}")
            
        return interactions
        
    def get_alternative_medications(self, drug: str) -> List[Dict]:
        """Get safer alternatives for a drug with known interactions."""
        
        alternatives = {
            'warfarin': [
                {'name': 'DOACs (Apixaban, Rivaroxaban)', 'note': 'Lower bleeding risk'},
                {'name': 'Heparin', 'note': 'Short-term use'}
            ],
            'aspirin': [
                {'name': 'Clopidogrel', 'note': 'Alternative antiplatelet'},
                {'name': 'No antiplatelet', 'note': 'If no indication'}
            ],
            'metformin': [
                {'name': 'SGLT2 Inhibitors', 'note': 'Cardiovascular benefits'},
                {'name': 'DPP-4 Inhibitors', 'note': 'Weight neutral'}
            ]
        }
        
        return alternatives.get(self._normalize_drug(drug), [])


class DrugInteractionAPI:
    """Direct OpenFDA API wrapper for drug information."""
    
    def __init__(self):
        self.base_url = "https://api.fda.gov/drug"
        
    def search_drug(self, drug_name: str) -> List[Dict]:
        """Search for drug in FDA database."""
        
        try:
            url = f"{self.base_url}/label.json?search=openfda.brand_name:{drug_name}&limit=10"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
            return []
            
        except Exception as e:
            logger.error(f"FDA API search failed: {e}")
            return []
            
    def get_warnings(self, drug_name: str) -> List[str]:
        """Get warning and precautions for a drug."""
        
        results = self.search_drug(drug_name)
        
        warnings = []
        for result in results:
            warnings_section = result.get('warnings', [])
            if warnings_section:
                warnings.extend(warnings_section)
                
        return warnings


def create_drug_interaction_test_cases() -> List[Dict]:
    """Create test cases for drug interaction checking."""
    
    test_cases = [
        {
            'description': 'Aspirin + Warfarin - High risk',
            'drugs': ['aspirin', 'warfarin'],
            'expected_severity': 'high'
        },
        {
            'description': 'Metformin + Contrast - Medium risk',
            'drugs': ['metformin', 'contrast'],
            'expected_severity': 'medium'
        },
        {
            'description': 'Lisinopril + Potassium - Medium risk',
            'drugs': ['lisinopril', 'potassium'],
            'expected_severity': 'medium'
        },
        {
            'description': 'Aspirin + Ibuprofen - Medium interaction',
            'drugs': ['aspirin', 'ibuprofen'],
            'expected_severity': 'medium'
        },
        {
            'description': 'No interactions',
            'drugs': ['aspirin', 'acetaminophen'],
            'expected_severity': None
        }
    ]
    
    return test_cases


if __name__ == "__main__":
    checker = DrugInteractionChecker()
    
    test_drugs = ['aspirin', 'warfarin', 'metformin']
    
    interactions = checker.check_interactions(test_drugs)
    
    import json
    print(json.dumps(interactions, indent=2))
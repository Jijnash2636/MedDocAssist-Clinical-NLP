# Clinical Note AI System
# IEEE Research Project

from .main import ClinicalNoteAI, ClinicalTextNormalizer, PHIDeidentifier, ConfidenceGate, OutputFormatter
from .ner_biobert import MedicalNER
from .summarizer import ClinicalSummarizer
from .icd_mapper import ICDMapper
from .drug_interaction import DrugInteractionChecker
from .multimodal_input import MultimodalInputHandler

__version__ = "1.0.0"
__all__ = [
    'ClinicalNoteAI',
    'ClinicalTextNormalizer',
    'PHIDeidentifier',
    'MedicalNER',
    'ClinicalSummarizer',
    'ICDMapper',
    'DrugInteractionChecker',
    'ConfidenceGate',
    'OutputFormatter',
    'MultimodalInputHandler'
]
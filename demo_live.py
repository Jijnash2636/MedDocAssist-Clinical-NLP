"""
MedDocAssist - Live Demo for Project Presentation
Multimodal Clinical NLP System: Text, Audio, Image inputs
"""

import gradio as gr
import sys
import os
import json
sys.path.insert(0, '.')

from src.main import ClinicalTextNormalizer, PHIDeidentifier
from src.multimodal_input import process_audio, process_image

normalizer = ClinicalTextNormalizer()
deidentifier = PHIDeidentifier()

SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")

# Rule-based ICD mapping
ICD_MAP = {
    "chest pain": ("R07.9", "Chest pain, unspecified"),
    "shortness of breath": ("R06.02", "Shortness of breath"),
    "hypertension": ("I10", "Essential (primary) hypertension"),
    "diabetes": ("E11.9", "Type 2 diabetes mellitus without complications"),
    "abdominal pain": ("R10.9", "Unspecified abdominal pain"),
    "headache": ("R51.9", "Headache, unspecified"),
    "fever": ("R50.9", "Fever, unspecified"),
}

VALID_PROBLEMS = [
    "chest pain", "shortness of breath", "hypertension", "diabetes",
    "abdominal pain", "headache", "fever", "nausea", "vomiting",
    "cough", "fatigue", "dizziness", "back pain", "joint pain",
    "asthma", "pneumonia", "infection", "allergy", "anxiety",
    "depression", "obesity", "arthritis", "anemia"
]

WELLNESS_KEYWORDS = [
    "healthy", "normal", "no complaints", "no issues", "no problems",
    "no health issues", "no acute issues", "no active complaints",
    "wellness check", "routine checkup", "asymptomatic", "stable condition"
]

REMOVE_WORDS = ["pain", "blood pressure", "patient", "history", "presents"]

DDI_PAIRS = [
    ("aspirin", "ibuprofen"),
    ("warfarin", "aspirin"),
    ("metformin", "contrast"),
    ("lisinopril", "potassium"),
]

TEST_MAP = {
    "ecg": "ECG",
    "electrocardiogram": "ECG",
    "cbc": "CBC",
    "complete blood count": "CBC",
    "cmp": "CMP",
    "comprehensive metabolic panel": "CMP",
    "ct scan": "CT Scan",
    "mri": "MRI",
    "x-ray": "X-Ray",
    "wbc": "WBC",
    "hba1c": "HbA1c",
    "blood test": "Blood Test",
    "urine test": "Urine Test",
}

SAMPLE_NOTES = {
    "Cardiac Patient": "Patient John Smith presents with chest pain and shortness of breath. BP is 140/90. History of hypertension and diabetes. Prescribed aspirin 81mg daily and metoprolol 25mg BID. ECG shows normal sinus rhythm. CBC and CMP ordered.",
    "Diabetic Follow-up": "Pt Mary Johnson comes for follow-up visit. HBA1C: 7.8. Reports SOB after exertion. Continues metformin 500mg BID and insulin glargine 10 units at night. Pt complains of occasional HA. BP stable at 130/85.",
    "Emergency Case": "Patient Robert Davis brought to ER with severe abdominal pain and vomiting. Temp 101.2F. WBC count elevated. CT scan ordered. Administered IV morphine 4mg stat. Pt allergic to penicillin. NPO until further notice.",
}


def extract_entities(text):
    text_lower = text.lower()

    problems = [p for p in VALID_PROBLEMS if p in text_lower]
    problems = [p for p in problems if p not in REMOVE_WORDS]
    problems = list(dict.fromkeys(problems))

    med_keywords = [
        "aspirin", "metformin", "insulin", "morphine", "ibuprofen",
        "metoprolol", "warfarin", "lisinopril", "penicillin", "amoxicillin",
        "atorvastatin", "omeprazole", "hydrochlorothiazide", "acetaminophen",
    ]
    treatments = [med for med in med_keywords if med in text_lower]

    tests = []
    for keyword, label in TEST_MAP.items():
        if keyword in text_lower and label not in tests:
            tests.append(label)

    return {"problems": problems, "treatments": treatments, "tests": tests}


def map_to_icd(problems):
    icd_codes = []
    for p in problems:
        if p in ICD_MAP:
            code, desc = ICD_MAP[p]
            icd_codes.append({"code": code, "description": desc})

    seen = set()
    unique_icd = []
    for icd in icd_codes:
        if icd["code"] not in seen:
            seen.add(icd["code"])
            unique_icd.append(icd)

    return unique_icd


def check_drug_interactions(treatments):
    alerts = []
    for d1, d2 in DDI_PAIRS:
        if d1 in treatments and d2 in treatments:
            alerts.append({
                "drug_1": d1,
                "drug_2": d2,
                "severity": "HIGH" if d1 in ["warfarin", "metformin"] else "MODERATE",
                "recommendation": f"Monitor {d1}+{d2} interaction"
            })
    return alerts


def generate_summary(problems, treatments, tests, raw_text=""):
    """Generate clinical summary with wellness detection."""
    parts = []
    if problems:
        parts.append(f"Patient presents with {', '.join(problems)}")
    if treatments:
        parts.append(f"Prescribed {', '.join(treatments)}")
    if tests:
        parts.append(f"Tests ordered: {', '.join(tests)}")
    
    if parts:
        return ". ".join(parts) + "."
    
    # No clinical entities found - check for wellness/normal language
    if raw_text:
        text_lower = raw_text.lower()
        if any(kw in text_lower for kw in WELLNESS_KEYWORDS):
            return "Patient presents with no acute complaints. No clinically significant findings identified."
    
    return "Clinical note processed. No active medical conditions identified."


def process_audio_file(audio_path):
    """Process uploaded audio file using Whisper ASR with fallback."""
    print("DEBUG - process_audio_file CALLED")
    if not audio_path or not os.path.exists(str(audio_path)):
        return "No audio file uploaded."
    
    # Try real Whisper transcription first
    try:
        print("DEBUG - Attempting Whisper transcription...")
        print(f"DEBUG - Audio path: {audio_path}")
        text = process_audio(audio_path)
        if text and text.strip():
            print("DEBUG - Whisper transcription successful")
            return text.strip()
    except Exception as e:
        print(f"DEBUG - Whisper failed: {e}")
    
    # Safe Fallback: Explicit failure message for demo honesty
    # If Whisper is not installed or fails, we show this instead of fake data.
    print("DEBUG - Using fallback (Whisper unavailable)")
    return "[WHISPER FAILED] Audio transcription unavailable. Please install OpenAI Whisper or use a supported audio format."


def process_image_file(image_path):
    """Process uploaded image file using Tesseract OCR with fallback."""
    print("DEBUG - process_image_file CALLED")
    if not image_path or not os.path.exists(str(image_path)):
        return "No image file uploaded."
    
    # Try real Tesseract OCR first
    try:
        print("DEBUG - Attempting Tesseract OCR...")
        text = process_image(image_path)
        if text and text.strip():
            print("DEBUG - Tesseract OCR successful")
            return text.strip()
    except Exception as e:
        print(f"DEBUG - Tesseract failed: {e}")
    
    # Safe Fallback: Explicit failure message for demo honesty
    # Tesseract is not installed on this system.
    print("DEBUG - Using fallback (Tesseract unavailable)")
    return "[OCR EXTRACTION UNAVAILABLE] Tesseract OCR engine not found. Please install Tesseract-OCR to enable real text extraction from images."


def analyze(input_mode, text_input, sample_note, audio_input, image_input):
    """Main analysis function - multimodal pipeline."""
    
    # DEBUG: Show what inputs are actually received
    print(f"DEBUG - Mode: {input_mode}")
    print(f"DEBUG - Text input: '{text_input[:50] if text_input else None}...'")
    print(f"DEBUG - Audio input: {audio_input}")
    print(f"DEBUG - Image input: {image_input}")

    source = ""
    raw = ""

    # STRICT ROUTING BASED ON SELECTED MODE
    # This prevents stale textbox values from overriding file uploads
    if input_mode == "Audio":
        print("DEBUG - Mode is AUDIO")
        if audio_input is not None:
            raw = process_audio_file(audio_input)
            source = "Audio (Whisper)"
        else:
            raw = "No audio file uploaded. Please upload an audio file in the Audio tab."
            source = "Audio (Error)"
            
    elif input_mode == "Image":
        print("DEBUG - Mode is IMAGE")
        if image_input is not None:
            raw = process_image_file(image_input)
            source = "Image (OCR)"
        else:
            raw = "No image file uploaded. Please upload an image in the Image tab."
            source = "Image (Error)"
            
    elif input_mode == "Text":
        print("DEBUG - Mode is TEXT")
        if text_input and text_input.strip():
            raw = text_input
            source = "Direct text input"
        elif sample_note and sample_note in SAMPLE_NOTES:
            raw = SAMPLE_NOTES[sample_note]
            source = f"Sample: {sample_note}"
        else:
            raw = ""
            source = "No text input"
    else:
        raw = ""
        source = "Unknown mode"

    print(f"DEBUG - Selected source: {source}")
    print(f"DEBUG - Raw text length: {len(raw)} chars")

    if not raw or not raw.strip() or "Error" in source or "No input" in source:
        error_msg = raw if "Error" in source or "No input" in source else "No input provided"
        return [
            error_msg, "No input", "No input",
            "No entities found", "No summary", "No ICD codes",
            "No drug interactions", "", {}, error_msg
        ]

    normalized = normalizer.normalize(raw)
    phi_removed = deidentifier.deidentify(normalized)
    entities = extract_entities(normalized)
    problems = entities["problems"]
    treatments = entities["treatments"]
    tests = entities["tests"]

    ner_parts = []
    if problems:
        ner_parts.append("**PROBLEMS:**\n" + "\n".join([f"- {p}" for p in problems]))
    if treatments:
        ner_parts.append("**TREATMENTS:**\n" + "\n".join([f"- {t}" for t in treatments]))
    if tests:
        ner_parts.append("**TESTS:**\n" + "\n".join([f"- {t}" for t in tests]))
    ner_text = "\n\n".join(ner_parts) if ner_parts else "No entities found"

    icd_codes = map_to_icd(problems)
    icd_text = ""
    for icd in icd_codes:
        icd_text += f"- **{icd['code']}** {icd['description']}\n"
    if not icd_text:
        icd_text = "No ICD codes mapped"

    summary = generate_summary(problems, treatments, tests, raw)

    drug_alerts = check_drug_interactions(treatments)
    drug_text = ""
    for alert in drug_alerts:
        drug_text += f"- **{alert['drug_1']}** + **{alert['drug_2']}**: [{alert['severity']}] {alert['recommendation']}\n"
    if not drug_text:
        drug_text = "No drug interactions detected"

    # Confidence Gate Logic
    if len(icd_codes) > 0:
        confidence = 0.85
        status = "AUTO-APPROVED"
    else:
        # No ICD codes - check for wellness/normal language
        text_lower = raw.lower()
        if any(kw in text_lower for kw in WELLNESS_KEYWORDS):
            confidence = 0.70  # Higher confidence for wellness notes (still REVIEW NEEDED)
            status = "REVIEW NEEDED"
        else:
            confidence = 0.60
            status = "REVIEW NEEDED"

    color = "#ff4444" if confidence < 0.75 else "#00ff88"
    conf_html = f"<span style='color:{color};font-weight:bold;font-size:16px'>{status}</span> (Score: {confidence:.2f})"

    json_out = {
        "source": source,
        "problems": problems,
        "treatments": treatments,
        "tests": tests,
        "icd_codes": [i["code"] for i in icd_codes],
        "drug_alerts": len(drug_alerts),
        "confidence": confidence,
        "status": status
    }

    label = f"Input: {source} | {len(raw)} chars"

    return [
        raw, normalized, phi_removed, ner_text, summary,
        icd_text, drug_text, conf_html, json_out, label
    ]


with gr.Blocks(title="MedDocAssist - Multimodal Clinical NLP") as demo:
    gr.Markdown("# MedDocAssist — Clinical NLP System")
    gr.Markdown("AI-Powered Clinical Note Summarization & Coding Assistant | **Multimodal Input: Text, Audio, Image**")

    # Master Input Mode Selector
    input_mode = gr.Radio(
        ["Text", "Audio", "Image"], 
        value="Text", 
        label="1. Select Input Mode",
        interactive=True
    )

    # Input Components (Controlled by Mode)
    with gr.Row():
        text_input = gr.Textbox(
            label="Enter Clinical Note", lines=5,
            placeholder="Type or paste clinical text here...",
            visible=True
        )
        sample_dropdown = gr.Dropdown(
            choices=list(SAMPLE_NOTES.keys()),
            value="Cardiac Patient", label="Or Select a Sample Note",
            visible=True
        )
    
    with gr.Row():
        audio_input = gr.Audio(
            sources=["upload", "microphone"], type="filepath",
            label="Audio Input",
            visible=False
        )
    
    with gr.Row():
        image_input = gr.Image(
            type="filepath", label="Image Input",
            visible=False
        )

    analyze_btn = gr.Button("2. Analyze Clinical Note", variant="primary", size="lg")

    # Mode switching logic to show/hide inputs and clear stale data
    def toggle_inputs(mode):
        if mode == "Text":
            # Show text inputs, hide/clear audio and image
            return [gr.update(visible=True), gr.update(visible=True), 
                    gr.update(visible=False, value=None), gr.update(visible=False, value=None)]
        elif mode == "Audio":
            # Show audio, hide/clear text and image
            return [gr.update(visible=False, value=""), gr.update(visible=False), 
                    gr.update(visible=True), gr.update(visible=False, value=None)]
        elif mode == "Image":
            # Show image, hide/clear text and audio
            return [gr.update(visible=False, value=""), gr.update(visible=False), 
                    gr.update(visible=False, value=None), gr.update(visible=True)]
        return [gr.update(visible=True), gr.update(visible=True), 
                gr.update(visible=False), gr.update(visible=False)]

    input_mode.change(
        toggle_inputs, 
        inputs=[input_mode], 
        outputs=[text_input, sample_dropdown, audio_input, image_input]
    )

    gr.Markdown("---\n## Pipeline Results")
    input_label = gr.Markdown("")

    with gr.Row():
        extracted_out = gr.Textbox(label="Extracted Raw Text", lines=3, interactive=False)
        norm_out = gr.Textbox(label="1. Normalized Text", lines=3, interactive=False)

    with gr.Row():
        phi_out = gr.Textbox(label="2. PHI Removed", lines=3, interactive=False)
        ner_out = gr.Markdown(label="3. Named Entities")

    with gr.Row():
        summary_out = gr.Textbox(label="4. Summary", lines=2, interactive=False)
        icd_out = gr.Markdown(label="5. ICD-10 Codes")

    with gr.Row():
        drug_out = gr.Markdown(label="6. Drug Interactions")
        conf_out = gr.Markdown(label="7. Confidence Gate")

    json_out = gr.JSON(label="JSON Output")

    gr.Markdown("---\n### Pipeline: Input → Normalization → PHI Removal → NER → Summarization → ICD-10 Mapping → Drug Check → Confidence Gate")

    analyze_btn.click(
        analyze,
        inputs=[input_mode, text_input, sample_dropdown, audio_input, image_input],
        outputs=[
            extracted_out, norm_out, phi_out, ner_out, summary_out,
            icd_out, drug_out, conf_out, json_out, input_label
        ]
    )


if __name__ == "__main__":
    print("=" * 50)
    print("  MedDocAssist - Multimodal Clinical NLP")
    print("  http://127.0.0.1:7864/")
    print("=" * 50)
    demo.queue(default_concurrency_limit=1).launch(
        server_port=7864, 
        server_name="0.0.0.0",
        show_error=True
    )

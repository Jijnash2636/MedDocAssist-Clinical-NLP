"""
Clinical Note AI - Simple Working Version
"""

import gradio as gr
import sys
sys.path.insert(0, '.')

from src.main import ClinicalNoteAI, ClinicalTextNormalizer, PHIDeidentifier
from src.multimodal_input import simulate_audio_transcript, simulate_clinical_image_text

system = ClinicalNoteAI()
normalizer = ClinicalTextNormalizer()
deidentifier = PHIDeidentifier()

def process(input_type, text_val, audio_val, img_val):
    try:
        # Debug output
        print(f"process called: type={input_type}, text={text_val[:50] if text_val else None}")
        
        if input_type == "text":
            raw = text_val or ""
        elif input_type == "audio":
            raw = simulate_audio_transcript() if not audio_val else "Audio file processed"
        else:
            raw = simulate_clinical_image_text() if not img_val else "Image processed"
        
        if not raw.strip():
            return ["No input"] * 9
        
        normalized = normalizer.normalize(raw)
        result = system.process_text(normalized)
        phi = deidentifier.deidentify(normalized)
        
        ents = result.get('entities', {})
        ner = f"Problems: {', '.join(ents.get('problems', [])[:3])}\nTreatments: {', '.join(ents.get('treatments', [])[:3])}"
        
        summary = result.get('summary', 'N/A')
        if summary and summary[0].islower():
            summary = summary[0].upper() + summary[1:]
        
        icds = "\n".join([f"{i['icd_code']} - {i['icd_description']}" for i in result.get('icd_codes', [])[:3]])
        
        drugs = "\n".join([f"{d['drug_1']}+{d['drug_2']}: {d['severity']}" for d in result.get('drug_interactions', [])[:3]]) or "None"
        
        conf = result.get('confidence', {})
        conf_str = f"Score: {conf.get('confidence', 0):.2f} | {'ACCEPTED' if not conf.get('requires_human_review') else 'REVIEW'}"
        
        j = {"input": input_type, "summary": summary, "codes": result.get('icd_codes', [])[:3]}
        
        return [normalized, phi, ner, summary, icds, drugs, conf_str, j, f"Source: {input_type.upper()}"]
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return [f"Error: {str(e)}"] * 9

# Simple interface
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 MedDocAssist - Clinical Note AI")
    
    with gr.Row():
        with gr.Column(scale=1):
            inp_type = gr.Radio(["text", "audio", "image"], label="Input Type", value="text")
        with gr.Column(scale=4):
            txt_in = gr.Textbox(label="Clinical Note", lines=3, visible=True)
            aud_in = gr.Audio(label="Audio", visible=False)
            img_in = gr.Image(label="Image", visible=False)
    
    btn = gr.Button("🔍 Analyze", variant="primary")
    
    with gr.Row():
        norm_out = gr.Textbox(label="Normalized", lines=2)
        phi_out = gr.Textbox(label="PHI Removed", lines=2)
    
    with gr.Row():
        ner_out = gr.Textbox(label="Entities", lines=3)
        sum_out = gr.Textbox(label="Summary", lines=3)
    
    with gr.Row():
        icd_out = gr.Textbox(label="ICD Codes", lines=2)
        drug_out = gr.Textbox(label="Drugs", lines=2)
    
    conf_out = gr.Textbox(label="Confidence", lines=1)
    json_out = gr.JSON(label="JSON")
    src_out = gr.Textbox(label="Source", lines=1)
    
    # Events
    def show_input(typ):
        return [typ=="text", typ=="audio", typ=="image"]
    
    inp_type.change(show_input, inp_type, [txt_in, aud_in, img_in])
    
    btn.click(process, [inp_type, txt_in, aud_in, img_in], 
              [norm_out, phi_out, ner_out, sum_out, icd_out, drug_out, conf_out, json_out, src_out])

if __name__ == "__main__":
    demo.launch(server_port=7864, server_name="0.0.0.0")
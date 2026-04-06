"""
Clinical Note AI - Clean Pipeline UI Design (Final)
"""

import gradio as gr
import sys
sys.path.insert(0, '.')

from src.main import ClinicalNoteAI, ClinicalTextNormalizer, PHIDeidentifier
from src.multimodal_input import simulate_audio_transcript, simulate_clinical_image_text

system = ClinicalNoteAI()
normalizer = ClinicalTextNormalizer()
deidentifier = PHIDeidentifier()

def analyze(input_type, text_in, audio_in, image_in):
    print(f"DEBUG: input_type={input_type}, text={text_in[:30] if text_in else None}, audio={audio_in}, img={type(image_in)}")
    try:
        if input_type == "text":
            raw = text_in or ""
        elif input_type == "audio":
            if audio_in:
                try:
                    from src.multimodal_input import process_audio
                    raw = process_audio(audio_in)
                except:
                    raw = simulate_audio_transcript()
            else:
                raw = simulate_audio_transcript()
        else:  # image
            try:
                from src.multimodal_input import process_image
                import os
                import numpy as np
                
                # Handle different image input types from Gradio
                if image_in is None:
                    raw = simulate_clinical_image_text()
                elif isinstance(image_in, str):
                    # It's a file path
                    raw = process_image(image_in)
                elif isinstance(image_in, (list, tuple)):
                    # List/tuple of data
                    raw = simulate_clinical_image_text()
                else:
                    # Try as numpy array
                    try:
                        img_array = np.array(image_in)
                        temp_path = os.path.join(os.path.dirname(__file__), 'temp_image.png')
                        from PIL import Image
                        Image.fromarray(img_array).save(temp_path)
                        raw = process_image(temp_path)
                    except:
                        raw = simulate_clinical_image_text()
            except Exception as e:
                print(f"Image error: {e}")
                raw = simulate_clinical_image_text()
        
        if not raw.strip():
            return "No input", "No input", "No entities", "No summary", "No codes", "No alerts", "Score: 0.00\nStatus: No input", {}, "Source: None"
        
        normalized = normalizer.normalize(raw)
        result = system.process_text(normalized)
        phi_display = deidentifier.deidentify(normalized)
        
        ents = result.get('entities', {})
        probs = ents.get('problems', [])
        treats = ents.get('treatments', [])
        tsts = ents.get('tests', [])
        
        probs_out = "Problems: " + (", ".join(probs) if probs else "None")
        treats_out = "Treatments: " + (", ".join(treats) if treats else "None")
        tsts_out = "Tests: " + (", ".join(tsts) if tsts else "None")
        ner_out = f"{probs_out}\n{treats_out}\n{tsts_out}"
        
        summary = result.get('summary', 'N/A')
        if summary and summary[0].islower():
            summary = summary[0].upper() + summary[1:]
        
        icd_list = result.get('icd_codes', [])[:5]
        seen_codes = set()
        icds = ""
        for i in icd_list:
            code = i.get('icd_code', '')
            if code and code not in seen_codes:
                seen_codes.add(code)
                desc = i.get('icd_description', '')
                icds += f"{code} - {desc}\n"
        if not icds:
            icds = "No codes mapped"
        
        drugs = ""
        for d in result.get('drug_interactions', []):
            drugs += f"{d['drug_1']} + {d['drug_2']}: {d['severity'].upper()}\n"
        if not drugs:
            drugs = "No interactions"
        
        conf = result.get('confidence', {})
        score = conf.get('confidence', 0)
        status = "✅ ACCEPTED" if not conf.get('requires_human_review') else "⚠️ REVIEW NEEDED"
        conf_out = f"Score: {score:.2f}\nStatus: {status}"
        
        source_display = f"Source: {input_type.upper()}"
        
        jout = {
            "input_type": input_type,
            "entities": result.get('entities', {}),
            "summary": summary,
            "icd_codes": result.get('icd_codes', [])[:5],
            "drug_interactions": result.get('drug_interactions', []),
            "confidence": conf
        }
        
        return normalized, phi_display, ner_out, summary, icds, drugs, conf_out, jout, source_display
        
    except Exception as e:
        import traceback
        err_msg = f"Error: {str(e)}"
        print(err_msg)
        print(traceback.format_exc())
        return err_msg, err_msg, err_msg, err_msg, err_msg, err_msg, err_msg, {"error": str(e)}, f"Source: ERROR"

def update_visibility(input_type):
    return [input_type == "text", input_type == "audio", input_type == "image"]

# ===================== UI =====================
with gr.Blocks(title="MedDocAssist") as demo:
    
    # HEADER
    gr.Markdown("""
    <div style="text-align:center; padding: 20px; background: linear-gradient(90deg, #1e3a8a, #3b82f6); border-radius: 10px; margin-bottom: 20px;">
        <h1 style="color: white; margin: 0;">🧠 MedDocAssist</h1>
        <p style="color: #e0e7ff; margin: 5px 0;">Clinical Note Summarization & Coding System</p>
    </div>
    """)
    
    # ===== SECTION 1: INPUT =====
    gr.Markdown("### 📥 INPUT")
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            itype = gr.Radio(
                ["text", "audio", "image"], 
                label="Input Type",
                value="text"
            )
        with gr.Column(scale=4):
            txt = gr.Textbox(
                label="📝 Clinical Input", 
                lines=3, 
                placeholder="Enter clinical note (e.g., Patient has chest pain...)",
                visible=True
            )
            aud = gr.Audio(
                label="🎤 Upload Audio",
                type="filepath",
                sources=["upload"],
                visible=False
            )
            img = gr.Image(
                label="🖼️ Upload Image",
                type="numpy",
                visible=False
            )
    
    btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
    
    # Source display
    source_out = gr.Textbox(label="📡 Source", lines=1, interactive=False)
    
    # ===== SECTION 2: PREPROCESSING =====
    gr.Markdown("---")
    gr.Markdown("### 🧹 PREPROCESSING")
    
    with gr.Row():
        out_norm = gr.Textbox(label="Normalized Text", lines=2)
        out_phi = gr.Textbox(label="PHI Removed", lines=2)
    
    # ===== SECTION 3: AI ANALYSIS =====
    gr.Markdown("---")
    gr.Markdown("### 🤖 AI ANALYSIS")
    
    with gr.Row():
        out_ner = gr.Textbox(label="🧬 Entities", lines=4)
        out_sum = gr.Textbox(label="📝 Summary", lines=4)
    
    with gr.Row():
        out_icd = gr.Textbox(label="🏥 ICD-10 Codes", lines=3)
        out_drug = gr.Textbox(label="💊 Drug Alerts", lines=3)
    
    # ===== SECTION 4: CONFIDENCE =====
    gr.Markdown("---")
    gr.Markdown("### ⚠️ CONFIDENCE GATE")
    
    out_conf = gr.Textbox(label="Decision", lines=2)
    
    # ===== SECTION 5: JSON OUTPUT =====
    gr.Markdown("---")
    gr.Markdown("### 📤 JSON OUTPUT (EHR Ready)")
    out_json = gr.JSON()
    
    # ===== CONNECTIONS =====
    btn.click(
        fn=analyze, 
        inputs=[itype, txt, aud, img], 
        outputs=[out_norm, out_phi, out_ner, out_sum, out_icd, out_drug, out_conf, out_json, source_out]
    )
    
    itype.select(
        fn=update_visibility,
        inputs=itype,
        outputs=[txt, aud, img]
    )
    
    clear_btn = gr.Button("🗑️ Clear")
    clear_btn.click(
        fn=lambda: ("text", "", None, None, "", "", "", {}, "Source: -"),
        inputs=[],
        outputs=[itype, txt, aud, img, out_norm, out_phi, out_ner, out_sum, out_icd, out_drug, out_conf, out_json, source_out]
    )

if __name__ == "__main__":
    demo.launch(server_port=7863, share=True, server_name="0.0.0.0")
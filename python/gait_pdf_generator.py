import sys
import os
import json
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

class AdvancedPDF(FPDF):
    def header(self):
        self.set_fill_color(30, 41, 59)
        self.rect(0, 0, 210, 18, style="F")
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(255, 255, 255)
        self.set_xy(12, 4)
        self.cell(0, 10, "NEUROSTRATUM | Advanced Biomechanical Evaluation", ln=True, align="L")
        self.ln(12)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"NeuroStratum AI Screening Protocol - Confidential Report - Page {self.page_no()}", align="C")

def get_llm_summary(metrics_dict):
    clean_dict = {k: v for k, v in metrics_dict.items() if not k.endswith("Series")}
    prompt = f"""
Review these clinical gait metrics: {json.dumps(clean_dict)}
Write a brief professional summary. Never diagnose. Use "may indicate" and "consider consulting a physiotherapist". Every number must exist in metrics.
"""
    payload = {"model": "qwen2.5:3b", "prompt": prompt, "stream": False, "options": {"temperature": 0.2}}
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            return res.get("response", "").strip().encode("latin-1", "ignore").decode("latin-1")
    except Exception:
        return (
            f"Evaluation classification: {metrics_dict.get('classification', 'Normal')}. "
            f"Cadence is {metrics_dict.get('cadence', 0)} spm with a confidence score of {metrics_dict.get('confidenceScore', 0)}%. "
            f"Findings may indicate movement variations; consider consulting a physiotherapist."
        )

def build_pdf(csv_path, output_pdf="NeuroStratum_Gait_Report.pdf"):
    sys.path.append(os.path.dirname(__file__))
    from session_analyze import compute_metrics
    analysis = compute_metrics(csv_path)
    metrics = analysis["metrics"]
    summary_text = get_llm_summary(metrics)

    pdf = AdvancedPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title & Metadata Section
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 6, "Screening Classification & Summary", ln=True)

    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 5, f"Source Target: {os.path.basename(csv_path)} | Result: {metrics['classification']}", ln=True)
    pdf.ln(4)

    # Metrics Table
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(56, 189, 248)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(50, 7, "Biomechanical Metric", 1, 0, "C", True)
    pdf.cell(45, 7, "Recorded Value", 1, 0, "C", True)
    pdf.cell(45, 7, "Standard Range", 1, 0, "C", True)
    pdf.cell(50, 7, "Evaluation Status", 1, 1, "C", True)

    rows = [
        ("Cadence", f"{metrics['cadence']} spm", "100 - 120 spm", "Normal" if 95 <= metrics['cadence'] <= 130 else "Attention"),
        ("Step Length", f"{metrics['stepLength']} cm", "50 - 80 cm", "Normal" if metrics['stepLength'] >= 40 else "Attention"),
        ("Asymmetry", f"{metrics['asymmetry']} cm", "< 5.0 cm", "Normal" if metrics['asymmetry'] <= 5.0 else "Attention"),
        ("Tracking Confidence", f"{metrics['confidenceScore']}%", "> 70.0%", "Optimal" if metrics['confidenceScore'] >= 70 else "Low"),
    ]
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    for name, val, target, status in rows:
        pdf.cell(50, 6, name, 1, 0, "L")
        pdf.cell(45, 6, val, 1, 0, "C")
        pdf.cell(45, 6, target, 1, 0, "C")
        pdf.cell(50, 6, status, 1, 1, "C")
    pdf.ln(6)

    # LLM Clinical Summary Box
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 6, "AI Clinical Interpretation & Disclaimers", ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(50, 50, 50)
    pdf.multi_cell(0, 4.5, summary_text)
    pdf.ln(3)
    pdf.multi_cell(0, 4.5, "Disclaimer: NeuroStratum is a preliminary prototyping tool. It never provides formal medical diagnoses. Always consult a qualified physiotherapist or physician for clinical assessments.")
    pdf.ln(5)

    # Multi-Metric Progression Graph Generation
    plt.figure(figsize=(8.5, 3.2))
    plt.style.use("ggplot")
    
    if "cadenceSeries" in metrics:
        plt.plot(metrics["cadenceSeries"], label="Cadence (spm)", color="#ef4444", linewidth=2)
    if "stepLengthSeries" in metrics:
        plt.plot(metrics["stepLengthSeries"], label="Step Length (cm)", color="#38bdf8", linewidth=2)
    if "asymmetrySeries" in metrics:
        plt.plot(metrics["asymmetrySeries"], label="Asymmetry (cm)", color="#10b981", linewidth=2)

    plt.title("Multi-Metric Progression Over Screening Windows", fontsize=10, color="#1e293b", fontweight="bold")
    plt.xlabel("Frame Window Index", fontsize=8)
    plt.ylabel("Metric Magnitude", fontsize=8)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    plot_path = f"temp_plot_{os.path.basename(csv_path)}.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    pdf.image(plot_path, x=12, w=185)
    if os.path.exists(plot_path):
        os.remove(plot_path)

    pdf.output(output_pdf)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        build_pdf(sys.argv[1], sys.argv[2])
    elif len(sys.argv) > 1:
        build_pdf(sys.argv[1])
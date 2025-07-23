import streamlit as st
import requests
import os

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

def run():
    # st.set_page_config(page_title="Smart ATS Resume Evaluator", layout="centered")
    st.title("📄 Smart ATS Resume Evaluator (Offline + Free)")

    uploaded_file = st.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])
    job_role = st.text_input("Target Job Role", value="Software Engineer")

    if uploaded_file and st.button("Evaluate"):
        file_bytes = uploaded_file.getvalue()
        files = {"file": (uploaded_file.name, file_bytes, uploaded_file.type)}

        # --- Get resume text for preview ---
        resume_text = ""
        with st.spinner("Extracting resume text..."):
            try:
                resp = requests.post(f"{API_BASE}/parse-resume/", files=files, timeout=30)
                resp.raise_for_status()
                resume_text = resp.json().get("resume_text", "")
            except Exception as err:
                st.error(f"Failed to parse resume: {err}")

        if not resume_text:
            st.error("Could not extract text from resume.")
            return

        # --- Display preview ---
        with st.expander("📃 Show Extracted Resume Text", expanded=False):
            st.text_area("Resume Preview", resume_text, height=200)

        # --- Get ATS evaluation ---
        with st.spinner("Evaluating ATS score..."):
            try:
                resp2 = requests.post(
                    f"{API_BASE}/ats-score/", files=files, data={"job_role": job_role}, timeout=60
                )
                resp2.raise_for_status()
                result = resp2.json().get("ats_result", "")
            except Exception as err:
                st.error(f"Failed to get ATS result: {err}")
                return
            # --- Parse Gemini Result for Pretty Output ---
            summary, score, suggestions = "", "", ""
            import re
            summary_match = re.search(r"Summary:\s*(.*)", result)
            score_match = re.search(r"Score.*?:\s*(\d+)", result)
            suggestions_match = re.search(r"Suggestions?:\s*(.*)", result)
            if summary_match:
                summary = summary_match.group(1).strip()
            if score_match:
                score = int(score_match.group(1))
            if suggestions_match:
                suggestions = suggestions_match.group(1).strip()

            # --- ATS Score Gauge ---
            import plotly.graph_objects as go
            gauge_color = "#26a69a" if score and score >= 8 else ("#ffa726" if score and score >= 5 else "#ef5350")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = score if score else 0,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "ATS Score (out of 10)"},
                gauge = {
                    'axis': {'range': [0, 10]},
                    'bar': {'color': gauge_color},
                    'steps': [
                        {'range': [0, 5], 'color': '#ef5350'},
                        {'range': [5, 8], 'color': '#ffa726'},
                        {'range': [8, 10], 'color': '#26a69a'}
                    ],
                }
            ))
            fig.update_layout(height=250, margin=dict(l=30, r=30, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            # --- Pretty Info Boxes ---
            st.success(f"Summary: {summary}" if summary else "Summary not found.")
            st.info(f"Suggestions: {suggestions}" if suggestions else "No suggestions found.")

# Call the app
if __name__ == "__main__":
    run()

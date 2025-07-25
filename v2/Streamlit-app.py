import streamlit as st
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import os
import plotly.graph_objects as go

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Future Trail | Career Navigator", layout="wide", page_icon="🚀")

# --- Modern CSS Styling ---
st.markdown(
    """
    <style>
    body {background-color: #f5f7fa;}
    .main-title {
        font-size: 2.6em;
        font-weight: 700;
        color: #1a237e;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 8px;
        letter-spacing: 1px;
    }
    .section-header {
        font-size: 1.3em;
        color: #3949ab;
        font-weight: 600;
        margin-top: 1.2em;
        margin-bottom: 0.5em;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3949ab 0%, #00c6ff 100%);
        color: white;
        border-radius: 8px;
        height: 2.8em;
        font-size: 1.1em;
        font-weight: 600;
        box-shadow: 0 2px 8px rgba(33,150,243,0.08);
        border: none;
    }
    .stTextInput>div>input, .stNumberInput>div>input, .stSelectbox>div>div>div>input {
        border-radius: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Main Title ---
st.markdown('<div class="main-title">🚀 Future Trail | Career Navigator</div>', unsafe_allow_html=True)

# --- Sidebar Navigation ---
page = st.sidebar.radio("Functionalities", ["🎓 Career Predictor", "📄 ATS Resume Evaluator"])

# --- Career Predictor ---
def run_career_predictor():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, "saved-models")
    mlb_dict = joblib.load(os.path.join(MODEL_DIR, "mlb_dict.pkl"))
    model = joblib.load(os.path.join(MODEL_DIR, "career_model.pkl"))

    st.markdown("<div class='section-header'>📝 Profile Information</div>", unsafe_allow_html=True)
    left_spacer, main_col, right_spacer = st.columns([1,3,1])
    with main_col:
        multi_label_inputs = {}
        for col in mlb_dict.keys():
            options = mlb_dict[col].classes_
            selected = st.multiselect(f"{col.replace('_', ' ')}", options, help=f"Select your {col.replace('_', ' ')}")
            multi_label_inputs[col] = selected
        preferred_style = st.selectbox("Preferred Work Style", ["Remote", "Hybrid", "Onsite"], help="Where do you prefer to work?")
        problem_style = st.selectbox("Problem Solving Style", ["Analytical", "Creative", "Logical", "Experimental"], help="How do you approach problems?")
        masters = st.radio("Do you want to go for Masters?", ["Yes", "No"], horizontal=True)
        research = st.radio("Interested in Research?", ["Yes", "No"], horizontal=True)
        cgpa = st.slider("Current CGPA", 2.0, 10.0, 7.5, 0.1, help="Your latest CGPA")
        projects = st.number_input("Current Projects Count", min_value=0, step=1, help="How many projects have you done?")
        internships = st.number_input("Internship Duration (in months)", min_value=0, step=1, help="Total months of internship experience")

    st.markdown("<div class='section-header'>🎯 Career Prediction</div>", unsafe_allow_html=True)
    def prepare_input():
        feature_parts = []
        for col, mlb in mlb_dict.items():
            selected_values = multi_label_inputs[col]
            encoded = mlb.transform([selected_values])
            df = pd.DataFrame(encoded, columns=[f"{col}_{c}" for c in mlb.classes_])
            feature_parts.append(df)
        other_features = pd.DataFrame([{
            "Preferred_Work_Style": {"Remote": 0, "Hybrid": 1, "Onsite": 2}.get(preferred_style, 0),
            "Problem_Solving_Style": {"Analytical": 0, "Creative": 1, "Logical": 2, "Experimental": 3}.get(problem_style, 0),
            "Wants_to_Go_for_Masters": 1 if masters.lower() == "yes" else 0,
            "Interested_in_Research": 1 if research.lower() == "yes" else 0,
            "CGPA": cgpa,
            "Current_Projects_Count": projects,
            "Internship_Experience": internships
        }])
        feature_parts.append(other_features)
        final_input = pd.concat(feature_parts, axis=1)
        for col in model.feature_names_in_:
            if col not in final_input:
                final_input[col] = 0
        return final_input[model.feature_names_in_]

    if st.button("🔍 Predict My Career", use_container_width=True):
        input_df = prepare_input()
        features_dict = input_df.iloc[0].to_dict()
        with st.spinner("Getting recommendation..."):
            try:
                resp = requests.post(f"{API_BASE}/predict-career/", json=features_dict, timeout=30)
                resp.raise_for_status()
                career = resp.json().get("recommended_career", "Unknown")
                st.success(f"🎯 We recommend: **{career}**", icon="🎯")
            except Exception as err:
                st.error(f"Failed to get recommendation: {err}")
            st.markdown("<div style='margin-top:0.5em;'></div>", unsafe_allow_html=True)
            st.markdown("<b>Your Selections:</b>", unsafe_allow_html=True)
            for k, v in multi_label_inputs.items():
                st.markdown(f"- <span style='color:#1976d2'><b>{k.replace('_', ' ')}:</b></span> {', '.join(v) if v else 'None'}", unsafe_allow_html=True)
            st.markdown(f"- <b>Preferred Work Style:</b> {preferred_style}", unsafe_allow_html=True)
            st.markdown(f"- <b>Wants to go for Masters:</b> {masters}", unsafe_allow_html=True)
            st.markdown(f"- <b>Interested in Research:</b> {research}", unsafe_allow_html=True)
            st.markdown(f"- <b>CGPA:</b> {cgpa}", unsafe_allow_html=True)
            st.markdown(f"- <b>Projects Count:</b> {projects}", unsafe_allow_html=True)
            st.markdown(f"- <b>Internship Duration (months):</b> {internships}", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>🎓 Recommended Resources</div>", unsafe_allow_html=True)
    st.markdown("Get started with these resources for your top careers:")
    resource_map = {
        "Data Scientist": ["Coursera ML by Andrew Ng", "Kaggle Competitions", "fast.ai"],
        "Cybersecurity Analyst": ["TryHackMe", "HackTheBox", "Cybrary"],
        "DevOps Engineer": ["Docker Mastery", "Learn Kubernetes", "CI/CD with Jenkins"],
        "Software Developer (Backend)": ["System Design Primer", "LeetCode", "Build REST APIs"],
        "UI/UX Designer": ["Google UX Certification", "Figma Basics", "Design Thinking by IDEO"]
    }
    for role, resources in resource_map.items():
        with st.expander(f"{role} Resources", expanded=False):
            for item in resources:
                st.markdown(f"- {item}")

# --- ATS Resume Evaluator ---
def run_ats_evaluator():
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
            import re
            summary, score, suggestions = "", "", ""
            summary_match = re.search(r"Summary:\s*(.*)", result)
            score_match = re.search(r"Score.*?:\s*(\d+)", result)
            suggestions_match = re.search(r"Suggestions?:\s*(.*)", result)
            if summary_match:
                summary = summary_match.group(1).strip()
            if score_match:
                score = int(score_match.group(1))
            if suggestions_match:
                suggestions = suggestions_match.group(1).strip()
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
            st.success(f"Summary: {summary}" if summary else "Summary not found.")
            st.info(f"Suggestions: {suggestions}" if suggestions else "No suggestions found.")

# --- Main Routing ---
if page == "🎓 Career Predictor":
    st.markdown('<div class="section-header"></div>', unsafe_allow_html=True)
    run_career_predictor()
elif page == "📄 ATS Resume Evaluator":
    st.markdown('<div class="section-header"></div>', unsafe_allow_html=True)
    run_ats_evaluator()

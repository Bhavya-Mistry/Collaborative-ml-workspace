import streamlit as st
from app import ats_app, career_app

st.set_page_config(page_title="Career Navigator", layout="wide", page_icon="🚀")

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
st.markdown('<div class="main-title">🚀 AI-Powered Career Navigator</div>', unsafe_allow_html=True)

# --- Navigation Buttons ---
col1, col2 = st.columns([1, 1])
with col1:
    nav_career = st.button("🎓 Career Predictor", use_container_width=True)
with col2:
    nav_ats = st.button("📄 ATS Resume Evaluator", use_container_width=True)

# --- Section Logic ---
if ("nav" not in st.session_state) or (nav_career and not nav_ats):
    st.session_state["nav"] = "career"
elif nav_ats:
    st.session_state["nav"] = "ats"

if st.session_state["nav"] == "career":
    st.markdown('<div class="section-header">🎓 Career Prediction & Guidance</div>', unsafe_allow_html=True)
    career_app.run()
elif st.session_state["nav"] == "ats":
    st.markdown('<div class="section-header">📄 ATS Resume Evaluation</div>', unsafe_allow_html=True)
    ats_app.run()

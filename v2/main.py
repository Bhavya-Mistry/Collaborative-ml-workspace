from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from app.resume_parser import parse_resume
from app.prompts import build_ats_prompt
from app.gemini_handler import get_gemini_response
import joblib
import pandas as pd
import tempfile
import os

app = FastAPI(title="Career Navigator API")

# Enable CORS so the Streamlit front-end can talk to this API when served from a different port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and encoders for career prediction (on startup)
career_model = None
mlb_dict = None
label_encoder = None

@app.on_event("startup")
def load_models():
    global career_model, mlb_dict, label_encoder
    career_model = joblib.load(r"trained-models/careermodel.pkl")
    mlb_dict = joblib.load(r"trained-models/mlbdict.pkl")
    label_encoder = joblib.load(r"trained-models/labelencoder.pkl")

@app.get("/")
def root():
    return {"message": "Career Navigator FastAPI is running!"}

class _SyncUploadWrapper:
    """Wraps FastAPI's UploadFile content into a sync-style object expected by parse_resume."""
    def __init__(self, filename: str, data: bytes):
        self.name = filename  # streamlit uses .name, so we mimic that attr
        self._data = data
    def read(self):
        return self._data


@app.post("/parse-resume/")
async def parse_resume_endpoint(file: UploadFile = File(...)):
    """Extract text from uploaded resume (PDF/DOCX)."""
    try:
        raw = await file.read()  # bytes
        wrapper = _SyncUploadWrapper(file.filename, raw)
        text = parse_resume(wrapper)
        return {"resume_text": text}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ats-score/")
async def ats_score_endpoint(
    file: UploadFile = File(...),
    job_role: str = Form("Software Engineer")
):
    """Evaluate resume for ATS score and summary."""
    try:
        raw = await file.read()
        wrapper = _SyncUploadWrapper(file.filename, raw)
        resume_text = parse_resume(wrapper)
        prompt = build_ats_prompt(resume_text, job_role)
        result = get_gemini_response(prompt)
        return {"ats_result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict-career/")
def predict_career(
    features: dict
):
    """Predict career recommendations based on user features (expects JSON)."""
    try:
        # This expects the client to send all necessary features as a dict
        df = pd.DataFrame([features])
        # Log input columns and model expected features
        print(f"Input features columns: {df.columns.tolist()}")
        print(f"Model expected features: {career_model.feature_names_in_.tolist()}")
        # Check for missing columns and add them with 0
        for col in career_model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        # Reorder columns to match model input
        df = df[career_model.feature_names_in_]
        pred = career_model.predict(df)[0]
        label = label_encoder.inverse_transform([pred])[0]
        return {"recommended_career": label}
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Error in predict_career: {tb}")
        return JSONResponse(status_code=500, content={"error": str(e), "traceback": tb})

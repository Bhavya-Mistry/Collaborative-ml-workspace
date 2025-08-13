from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import traceback

# These would be in separate files in a real app
from app.resume_parser import parse_resume
from app.prompts import build_ats_prompt
from app.gemini_handler import get_gemini_response

app = FastAPI(title="Career Navigator API")

# --- Middleware ---
# In production, change "*" to your Streamlit app's URL for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Globals for ML Models ---
career_model = None
label_encoder = None

# --- Pydantic Models for Validation ---
# This model ensures the data from the frontend has the correct structure and types.
class CareerFeatures(BaseModel):
    CGPA: float
    Current_Projects_Count: int
    Internship_Experience: int
    Wants_to_Go_for_Masters: int
    Interested_in_Research: int
    # The frontend will send a dictionary that matches this structure.
    # We use a flexible dict here, but in a real app, you might list all possible features.
    # For this example, we'll assume the frontend sends all necessary columns.
    # The frontend is now responsible for ensuring all encoded columns are present.
    # This is a good example of how to define it for a dynamic set of features.
    # We will pass a dict from the frontend and load it directly.
    # Let's adjust the endpoint to accept a raw dict for simplicity, but with a clear note.
    pass # Pydantic model is best practice, but let's stick to the user's dict for now and just fix the logic.


class ATSRequest(BaseModel):
    resume_text: str
    job_role: str

# --- App Events ---
@app.on_event("startup")
def load_models():
    """Load models on startup to avoid loading them on every request."""
    global career_model, label_encoder
    career_model = joblib.load(r"trained-models/careermodel.pkl")
    label_encoder = joblib.load(r"trained-models/labelencoder.pkl")

# --- Helper Classes ---
class _SyncUploadWrapper:
    def __init__(self, filename: str, data: bytes):
        self.name = filename
        self._data = data
    def read(self):
        return self._data

# --- API Endpoints ---
@app.get("/")
def root():
    return {"message": "Career Navigator FastAPI is running!"}

@app.post("/parse-resume/")
async def parse_resume_endpoint(file: UploadFile = File(...)):
    """Extracts text from an uploaded resume file (PDF/DOCX)."""
    try:
        raw = await file.read()
        wrapper = _SyncUploadWrapper(file.filename, raw)
        text = parse_resume(wrapper)
        return {"resume_text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing resume: {str(e)}")
import traceback # Make sure this is imported at the top of your file

@app.post("/ats-score/")
async def ats_score_endpoint(request: ATSRequest):
    try:
        prompt = build_ats_prompt(request.resume_text, request.job_role)
        result = get_gemini_response(prompt)
        return {"ats_result": result}
    except Exception as e:
        # This will print the full, detailed error to your terminal
        print("--- ERROR IN /ats-score/ ---")
        traceback.print_exc()
        print("-----------------------------")
        raise HTTPException(status_code=500, detail=f"Error getting ATS score: {str(e)}")

@app.post("/predict-career/")
def predict_career_endpoint(features: dict):
    """
    Predicts a career based on a JSON object of user features.
    NOTE: Using a Pydantic model is strongly recommended here for production.
    """
    if not career_model:
        raise HTTPException(status_code=500, detail="Career model is not loaded.")
    try:
        # The frontend is now responsible for sending a complete dictionary.
        df = pd.DataFrame([features])
        
        # Ensure columns are in the same order as when the model was trained.
        model_cols = career_model.feature_names_in_
        # Add any missing columns with a value of 0
        for col in model_cols:
            if col not in df.columns:
                df[col] = 0
        
        # Reorder to match model's expected input
        df = df[model_cols]
        
        prediction = career_model.predict(df)
        label = label_encoder.inverse_transform(prediction)[0]
        return {"recommended_career": label}
    except Exception as e:
        tb = traceback.format_exc()
        # In production, log the traceback but don't send it to the client.
        print(f"Error in predict_career: {tb}") 
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
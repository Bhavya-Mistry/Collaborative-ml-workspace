import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load your .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Configure the Gemini API
genai.configure(api_key=api_key)

# ⚠ Use the lighter model to avoid quota issues
model = genai.GenerativeModel("models/gemini-2.5-flash")

def get_gemini_response(prompt):
    try:
        print(f"Using model: {model._model_name}")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

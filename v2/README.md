# Future Trail | Career Navigator

## 1. Project Overview

Career Navigator v2 is a comprehensive web application designed to assist users in their career development journey. It leverages machine learning and artificial intelligence to provide personalized career recommendations and resume feedback. The application is built with a modern tech stack, featuring a Streamlit frontend for an interactive user experience and a FastAPI backend for robust and scalable API services.

This document provides a detailed look into the application's features, architecture, and setup process.

## 2. Core Features

The application offers two primary functionalities accessible after user login:

### 2.1. 🎓 Career Predictor

This feature helps users discover potential career paths based on their academic background, skills, and personal preferences.

**How it works:**
-   Users input various details through an interactive form on the Streamlit frontend.
-   The frontend sends this data to the `/predict-career/` API endpoint.
-   The backend uses a pre-trained Random Forest Classifier to predict the most suitable career.
-   The recommended career is displayed to the user, along with a curated list of learning resources.

### 2.2. 📄 ATS Resume Evaluator

This tool provides an automated analysis of a user's resume against a specific job role, simulating an Applicant Tracking System (ATS).

**How it works:**
1.  **Resume Upload:** Users upload their resume in either PDF (`.pdf`) or Microsoft Word (`.docx`) format.
2.  **Text Parsing:** The `/parse-resume/` endpoint extracts the raw text from the document.
3.  **AI-Powered Evaluation:** The extracted text and a target job role are sent to the `/ats-score/` endpoint, which uses the Google Gemini API to generate an evaluation.
4.  **Results Display:** The application presents a detailed evaluation, including an ATS match score, a summary, and actionable suggestions for improvement.

## 3. Technical Architecture

### 3.1. Frontend (Streamlit)

-   **Framework:** Streamlit
-   **Description:** The frontend is a single-page application built with Streamlit. It provides a user-friendly interface for the two main features and handles user authentication.
-   **Key Components:**
    -   **`frontend.py`**: The main application script.
    -   **Authentication**: Uses `streamlit-authenticator` with a PostgreSQL backend to manage users. It includes login, logout, and registration forms.
    -   **Career Predictor Page**: A form with various input widgets (`st.multiselect`, `st.selectbox`, `st.slider`, `st.number_input`, `st.radio`) to collect user data.
    -   **ATS Resume Evaluator Page**: A file uploader (`st.file_uploader`) for resumes and a text input for the target job role. It displays the results using a Plotly gauge chart for the score and formatted markdown for the summary and suggestions.
-   **Communication:** Interacts with the backend via HTTP requests using the `requests` library.

### 3.2. Backend (FastAPI)

-   **Framework:** FastAPI
-   **Description:** The backend provides a RESTful API for the frontend. It handles the core logic of the application, including machine learning predictions and AI-powered evaluations.
-   **API Endpoints:**
    -   **`POST /predict-career/`**
        -   **Request Body:** A JSON object containing the user's features.
            ```json
            {
              "CGPA": 7.5,
              "Current_Projects_Count": 5,
              "Internship_Experience": 6,
              "Wants_to_Go_for_Masters": 1,
              "Interested_in_Research": 0,
              "Programming_Languages_Python": 1,
              "Interest_Areas_Data Science": 1,
              ...
            }
            ```
        -   **Response Body:** A JSON object with the recommended career.
            ```json
            {
              "recommended_career": "Data Scientist"
            }
            ```
    -   **`POST /parse-resume/`**
        -   **Request Body:** `multipart/form-data` with a file upload.
        -   **Response Body:** A JSON object with the extracted text.
            ```json
            {
              "resume_text": "..."
            }
            ```
    -   **`POST /ats-score/`**
        -   **Request Body:** A JSON object with the resume text and job role.
            ```json
            {
              "resume_text": "...",
              "job_role": "Software Engineer"
            }
            ```
        -   **Response Body:** A JSON object with the raw evaluation from the Gemini API.
            ```json
            {
              "ats_result": "- Summary: ...\n- Score (out of 10): ...\n- Suggestions: ..."
            }
            ```

### 3.3. Database

-   **Type:** PostgreSQL
-   **Schema:** A single table named `users` is used for authentication.
    ```sql
    CREATE TABLE users (
        id SERIAL PRIMARY KEY,
        name VARCHAR(255),
        username VARCHAR(255) UNIQUE,
        password VARCHAR(255),
        email VARCHAR(255)
    );
    ```
-   **Usage:** Stores user credentials. Passwords are hashed using `bcrypt`.

### 3.4. Artificial Intelligence & Machine Learning

-   **Career Prediction Model:**
    -   **Algorithm:** `RandomForestClassifier` from scikit-learn.
    -   **Training Data:** A CSV file (`19k.csv`) containing synthetic or real career data.
    -   **Features:** The model is trained on a combination of one-hot encoded and multi-label binarized features derived from user inputs.
        -   **Multi-label features**: `Programming_Languages`, `Certifications`, `Extracurricular_Interests`, `Interest_Areas`, `Soft_Skills`, `Tools_Techstack`, `Favourite_Subjects`, `Problem_Solving_Style`.
        -   **One-hot encoded feature**: `Preferred_Work_Style`.
        -   **Other features**: `CGPA`, `Current_Projects_Count`, `Internship_Experience`, `Wants_to_Go_for_Masters`, `Interested_in_Research`.
    -   **Saved Artifacts:** `careermodel.pkl`, `labelencoder.pkl`, `mlbdict.pkl`, `ohencoder.pkl`.
-   **Resume Evaluation (Gemini API):**
    -   **Model:** `gemini-2.5-flash`.
    -   **Prompt Engineering:** A specific prompt is constructed in `app/prompts.py` to instruct the Gemini model to act as an ATS, providing a structured response containing a summary, score, and suggestions.

## 4. Detailed Setup and Run Instructions

1.  **Prerequisites:**
    -   Python 3.8+
    -   PostgreSQL server

2.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd career-navigator/v2
    ```

3.  **Set up the Python environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

4.  **Set up the database:**
    -   Connect to your PostgreSQL server and create a new database.
    -   Create the `users` table using the SQL command from section 3.3.

5.  **Configure environment variables:**
    -   Create a `.env` file in the `career-navigator/v2` directory.
    -   Add your Google Gemini API key to the `.env` file:
        ```
        GEMINI_API_KEY="your_api_key_here"
        ```
    -   Configure Streamlit secrets for the database connection. Create a file at `~/.streamlit/secrets.toml` and add:
        ```toml
        [DB_CONNECTION_STRING]
        value = "postgresql://user:password@host:port/database"
        ```

6.  **Run the application:**
    -   **Backend:** Open a terminal and run the FastAPI server:
        ```bash
        uvicorn backend:app --reload
        ```
    -   **Frontend:** Open a second terminal and run the Streamlit app:
        ```bash
        streamlit run frontend.py
        ```

7.  **Access the application:**
    -   Open your web browser and go to `http://localhost:8501`.

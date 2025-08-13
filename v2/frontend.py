import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import os
import plotly.graph_objects as go
import re
import psycopg2 # You need to install this: pip install psycopg2-binary
import streamlit_authenticator as stauth

# --- DATABASE FUNCTIONS (Replaces YAML) ---
def get_db_connection():
    """Establishes a connection to the PostgreSQL database using secrets."""
    conn = psycopg2.connect(st.secrets["DB_CONNECTION_STRING"])
    return conn

# Updated fetch_users function to include email
def fetch_users():
    """Fetches user data from the database for the authenticator."""
    try:
        with get_db_connection() as conn: # Use 'with' for the connection
            with conn.cursor() as cur:     # And for the cursor
                cur.execute("SELECT name, username, password, email FROM users")
                users = cur.fetchall()

        credentials = {"usernames": {}}
        for name, username, password, email in users:
            credentials["usernames"][username] = {
                "name": name, 
                "password": password,
                "email": email
            }
        return credentials
    except (Exception, psycopg2.DatabaseError) as error:
        st.error(f"Error fetching users from database: {error}")
        return {"usernames": {}}

# Updated save_new_user function to include email
def save_new_user(username, name, hashed_password, email):
    """Saves a new registered user to the database."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (name, username, password, email) VALUES (%s, %s, %s, %s)",
                    (name, username, hashed_password, email)
                )
                conn.commit()
        return True
    except (Exception, psycopg2.DatabaseError) as error:
        # Rollback the transaction on error
        if 'conn' in locals() and conn is not None:
            conn.rollback()
        st.error(f"Error saving new user: {error}")
        return False

# --- USER AUTHENTICATION (Using Database) ---
# Fetch credentials from the database instead of a file
user_credentials = fetch_users()

# Create the config dictionary programmatically
config = {
    'credentials': user_credentials,
    'cookie': {
        'name': st.secrets.get("COOKIE_NAME", "some_cookie_name"),
        'key': st.secrets.get("COOKIE_KEY", "some_secret_key"),
        'expiry_days': 30
    }
}

# Create authenticator object
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Initialize session state for registration
if "registration_success" not in st.session_state:
    st.session_state.registration_success = False

# --- MAIN APP ---
st.set_page_config(page_title="Future Trail | Career Navigator", layout="wide", page_icon="🚀")

if st.session_state.get("authentication_status"):
    # --- LOGGED-IN VIEW ---
    st.title(f'Welcome Back {st.session_state["name"]}!')
    authenticator.logout('Logout', 'sidebar')
    
    API_BASE = os.getenv("API_BASE", "https://playground-pknf.onrender.com")

    # --- YOUR EXISTING UI AND ML LOGIC (UNCHANGED) ---
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
    resource_map = {
            "Software Developer (Backend)": [
                "System Design Primer (GitHub)",
                "LeetCode Practice",
                "Build REST APIs with Flask/Django (Udemy)"
            ],
            "Software Developer (Frontend)": [
                "Frontend Masters",
                "JavaScript: The Good Parts (Book)",
                "CSS-Tricks Tutorials"
            ],
            "Full Stack Developer": [
                "The Odin Project",
                "Full-Stack Web Dev with React Specialization (Coursera)",
                "Build Responsive Real‑World Websites with HTML & CSS (Udemy)"
            ],
            "Mobile Developer (Android)": [
                "Android Basics Nanodegree (Udacity)",
                "Android Developer Documentation (developer.android.com)",
                "Kotlin for Android Developers (Book)"
            ],
            "Mobile Developer (iOS)": [
                "iOS App Development with Swift Specialization (Coursera)",
                "Ray Wenderlich iOS Tutorials",
                "Hacking with Swift (Paul Hudson)"
            ],
            "Game Developer": [
                "Unity Learn",
                "Unreal Engine Online Learning",
                "Game Programming Patterns (Book)"
            ],
            "AR/VR Developer": [
                "Unity XR: VR Development (Udemy)",
                "Coursera AR & VR Specialization",
                "Google ARCore Developer Documentation"
            ],
            "Blockchain Developer": [
                "Ethereum and Solidity: The Complete Developer's Guide (Udemy)",
                "Blockchain Developer Nanodegree (Udacity)",
                "Mastering Bitcoin (Book)"
            ],
            "Data Scientist": [
                "Machine Learning by Andrew Ng (Coursera)",
                "Kaggle Competitions",
                "fast.ai Practical Deep Learning"
            ],
            "Machine Learning Engineer": [
                "Deep Learning Specialization (Coursera)",
                "ML Engineer Nanodegree (Udacity)",
                "Hands‑On Machine Learning with Scikit‑Learn, Keras & TensorFlow (Book)"
            ],
            "AI Prompt Engineer": [
                "Prompt Engineering for Everyone (DeepLearning.AI)",
                "The Art of Prompting (Blog series)",
                "OpenAI Cookbook Examples"
            ],
            "Big Data Engineer": [
                "Big Data Specialization (Coursera)",
                "Hadoop: The Definitive Guide (Book)",
                "Data Engineering on Google Cloud Platform (Coursera)"
            ],
            "MLOps Engineer": [
                "MLOps Fundamentals (Coursera)",
                "Kubeflow Documentation",
                "Building Machine Learning Pipelines (O'Reilly)"
            ],
            "AI Ethicist": [
                "Ethics of AI and Big Data (edX)",
                "AI Ethics by Oxford (Online)",
                "Weapons of Math Destruction (Book)"
            ],
            "Data Engineer": [
                "Data Engineering Nanodegree (Udacity)",
                "Designing Data‑Intensive Applications (Book)",
                "ETL Pipelines with Airflow (Udemy)"
            ],
            "DevOps Engineer": [
                "Docker Mastery (Udemy)",
                "Learn Kubernetes (Coursera)",
                "CI/CD with Jenkins (Pluralsight)"
            ],
            "Cloud Engineer": [
                "AWS Certified Solutions Architect – Associate (A Cloud Guru)",
                "Google Cloud Professional Cloud Architect (Coursera)",
                "Microsoft Certified: Azure Administrator (Udemy)"
            ],
            "Solutions Architect": [
                "AWS Certified Solutions Architect – Professional",
                "Architecting with Google Cloud (Coursera)",
                "Azure Solutions Architect Expert (Microsoft Learn)"
            ],
            "Platform Engineer": [
                "Platform Engineering with Terraform (Udemy)",
                "Kubernetes Operators (Book)",
                "GitOps with Flux (Weaveworks)"
            ],
            "Systems Engineer": [
                "Computer Systems: A Programmer's Perspective (Book)",
                "Linux System Administration (Udemy)",
                "Red Hat Certified Engineer (RHCE)"
            ],
            "Cybersecurity Analyst": [
                "TryHackMe",
                "Hack The Box",
                "Cybrary Security Analyst Path"
            ],
            "Security Engineer": [
                "Practical Network Penetration Tester (Udemy)",
                "Offensive Security Certified Professional (OSCP)",
                "Security Engineering (Book)"
            ],
            "Quality Assurance Engineer": [
                "ISTQB Foundation (Udemy)",
                "Selenium WebDriver with Java (Udemy)",
                "Software Testing: Principles and Practices (Book)"
            ],
            "Software Test Automation Engineer": [
                "Test Automation University",
                "Appium: Mobile Test Automation (Udemy)",
                "Robot Framework Guide (Online)"
            ],
            "Research Intern (AI/ML)": [
                "ArXiv.org Papers",
                "Google AI Residency Insights",
                "FastAI Kaggle Research"
            ],
            "Research Intern (General)": [
                "University Lab Websites",
                "ResearchGate Networking",
                "How to Write a Research Paper (Course)"
            ],
            "MS in Computer Science": [
                "Official University Program Pages",
                "GRE Prep Course (Magoosh)",
                "Statement of Purpose Guide (Online)"
            ],
            "MS in Data Science": [
                "Data Science MS Program Reviews (CSRankings)",
                "GRE/TOEFL Prep",
                "MSDS Statement of Purpose Samples"
            ],
            "Ph.D. Candidate": [
                "PhD Application Workshop (Coursera)",
                "Academic Writing for Graduate Students (Book)",
                "How to Choose a PhD Advisor (Blog)"
            ],
            "UI/UX Designer": [
                "Google UX Design Certificate (Coursera)",
                "Figma for Beginners (Udemy)",
                "Design Thinking by IDEO (Online)"
            ],
            "Product Manager": [
                "Product Management 101 (Udemy)",
                "Inspired (Book by Marty Cagan)",
                "One Month Product Management"
            ],
            "Project Manager": [
                "PMP Certification Prep (Udemy)",
                "Scrum Master Certification (Scrum.org)",
                "Project Management for Non‑Project Managers (LinkedIn Learning)"
            ],
            "Technical Writer": [
                "Technical Writing: How to Write Software Docs (Udemy)",
                "Google Developer Documentation Style Guide",
                "Docs Like Code (Book)"
            ],
            "Technical Consultant": [
                "Consulting Case Interview Prep (Coursera)",
                "The McKinsey Way (Book)",
                "Frameworks for Problem Solving (Online)"
            ],
            "Technical Program Manager (TPM)": [
                "TPM Interview Prep (YouTube)",
                "Cracking the PM Interview (Book)",
                "Program Management Professional (PgMP)"
            ],
            "Business Analyst": [
                "Business Analysis Fundamentals (Udemy)",
                "BABOK Guide (IIBA)",
                "Data Analytics with Excel (Coursera)"
            ],
            "Database Administrator": [
                "Oracle Certified Professional (OCP)",
                "Microsoft SQL Server Certification",
                "High Performance MySQL (Book)"
            ],
            "Network Engineer": [
                "CCNA Certification (Cisco)",
                "Juniper Networks Certification",
                "Network Warrior (Book)"
            ],
            "Embedded Systems Engineer": [
                "Embedded Systems Essentials with Arm (Coursera)",
                "Making Embedded Systems (Book)",
                "Microcontroller Projects (Udemy)"
            ],
            "IoT Engineer": [
                "Introduction to IoT (edX)",
                "Raspberry Pi Projects (Book)",
                "IoT Specialization (Coursera)"
            ],
            "Bioinformatics Engineer": [
                "Bioinformatics Specialization (Coursera)",
                "Bioconductor Workshops",
                "Algorithms on Strings, Trees, and Sequences (Book)"
            ]
        } 
    
    st.markdown('<div class="main-title">🚀 Future Trail | Career Navigator</div>', unsafe_allow_html=True)
    page = st.sidebar.radio("Functionalities", ["🎓 Career Predictor", "📄 ATS Resume Evaluator"])

    def run_career_predictor():
        career = None
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        MODEL_DIR = os.path.join(BASE_DIR, "trained-models")
        mlb_dict = joblib.load(os.path.join(MODEL_DIR, "mlbdict.pkl"))
        ohe = joblib.load(os.path.join(MODEL_DIR, "ohencoder.pkl"))

        st.markdown("<div class='section-header'>📝 Profile Information</div>", unsafe_allow_html=True)
        left_spacer, main_col, right_spacer = st.columns([1,3,1])
        with main_col:
            multi_label_inputs = {}
            for col in mlb_dict.keys():
                options = mlb_dict[col].classes_
                selected = st.multiselect(f"{col.replace('_', ' ')}", options, help=f"Select your {col.replace('_', ' ')}")
                multi_label_inputs[col] = selected
            preferred_style = st.selectbox("Preferred Work Style", ohe.categories_[0], help="Where do you prefer to work?")
            problem_style = st.multiselect("Problem Solving Style", mlb_dict['Problem_Solving_Style'].classes_, help="How do you approach problems?")
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
                # One-hot encode Preferred_Work_Style using ohe.categories_
                **{f"Preferred_Work_Style_{cls}": 1 if cls == preferred_style else 0 for cls in ohe.categories_[0]},
                # Multi-label encode Problem_Solving_Style
                **dict(zip([f"Problem_Solving_Style_{cls}" for cls in mlb_dict['Problem_Solving_Style'].classes_], 
                             mlb_dict['Problem_Solving_Style'].transform([problem_style])[0])),
                "Wants_to_Go_for_Masters": 1 if masters.lower() == "yes" else 0,
                "Interested_in_Research": 1 if research.lower() == "yes" else 0,
                "CGPA": cgpa,
                "Current_Projects_Count": projects,
                "Internship_Experience": internships
            }])
            feature_parts.append(other_features)
            final_input = pd.concat(feature_parts, axis=1)
            # Align columns with mlb_dict keys and other features
            # No model loaded locally, so just return all columns
            return final_input

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

        if career:
            recommended_resources = resource_map.get(career, [])
            if not recommended_resources:
                st.warning("😕 Oops! We don't have curated resources for that exact role yet.")
            else:
                for item in recommended_resources:
                    st.markdown(f"- {item}")

    def run_ats_evaluator():
        st.title("📄 Smart ATS Resume Evaluator ")
        uploaded_file = st.file_uploader("Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])
        job_role = st.text_input("Target Job Role", value="Data Science", help="Enter the job role you are targeting for ATS evaluation.")

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
        # 1. Create the JSON payload
                    ats_payload = {
                        "resume_text": resume_text, # Use the text you got from /parse-resume/
                        "job_role": job_role
                    }
            
                    # 2. Send the request with `json=` instead of `files=`
                    resp2 = requests.post(
                        f"{API_BASE}/ats-score/",
                        json=ats_payload,  # <--- THE FIX IS HERE
                        timeout=60
                    )
                    # ... rest of the code

                    resp2.raise_for_status()
                    result = resp2.json().get("ats_result", "")
                except Exception as err:
                    st.error(f"Failed to get ATS result: {err}")
                    return

            with st.expander("🕵️ Click to see Raw API Response for Debugging"):
                st.text_area("API 'result' string:", result, height=100)

            # Parse the result string using a more flexible regex to handle optional markdown bolding
            summary_match = re.search(r"-\s*(?:\*\*)?Summary(?:\*\*)?:\s*(.*)", result, re.DOTALL | re.IGNORECASE)
            score_match = re.search(r"-\s*(?:\*\*)?Score \(out of 10\)(?:\*\*)?:\s*(\d+\.?\d*)", result, re.IGNORECASE)
            suggestions_match = re.search(r"-\s*(?:\*\*)?Suggestions(?:\*\*)?:\s*(.*)", result, re.DOTALL | re.IGNORECASE)

            # Check if parsing was successful. If not, show a clear error and stop.
            if not score_match or not summary_match:
                st.error(
                    "❌ **Error Parsing API Response**\n\n"
                    "The application could not find a valid score or summary in the response from the server. "
                    "Please check the raw API response in the expander above to diagnose the issue with your backend."
                )
                return  # Stop further execution

            summary = summary_match.group(1).strip()
            score = float(score_match.group(1))
            suggestions_str = suggestions_match.group(1).strip() if suggestions_match else ""

            st.markdown("---")
            st.markdown("<h2 style='text-align: center; color: #3949ab;'>Your ATS Evaluation Results</h2>", unsafe_allow_html=True)
            st.markdown("<div style='margin-top:1.5em;'></div>", unsafe_allow_html=True)

            col1, col2 = st.columns([2, 3])

            with col1:
                gauge_color = "#4caf50" if score >= 8 else ("#ff9800" if score >= 5 else "#f44336")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "<b>ATS Match Score</b>", 'font': {'size': 20, 'color': '#1a237e'}},
                    gauge={
                        'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': gauge_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "#d1d1d1",
                        'steps': [
                            {'range': [0, 5], 'color': '#ffebee'},
                            {'range': [5, 8], 'color': '#fff3e0'},
                            {'range': [8, 10], 'color': '#e8f5e9'},
                        ],
                    }
                ))
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font={'color': "#3949ab", 'family': "Arial, sans-serif"},
                    height=280,
                    margin=dict(l=10, r=10, t=50, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("<h3 style='color: #1a237e;'>📝 Evaluation Summary</h3>", unsafe_allow_html=True)
                if summary:
                    st.info(summary)
                else:
                    st.warning("Could not generate a summary for this resume.")

            st.markdown("<div style='margin-top:1.5em;'></div>", unsafe_allow_html=True)
            st.markdown("---")

            st.markdown("<h3 style='color: #1a237e;'>💡 Actionable Suggestions to Improve Your Score</h3>", unsafe_allow_html=True)

            # --- START OF IMPROVED SUGGESTIONS LOGIC ---
            if suggestions_str:
                suggestions_list = []
                # Check for standard markdown list formats (e.g., "- item\n- item")
                if '\n-' in suggestions_str or '\n*' in suggestions_str:
                    # Split by newlines and clean up list markers
                    suggestions_list = [
                        s.strip('*-. ') for s in suggestions_str.split('\n') if s.strip()
                    ]
                else:
                    # If no list format is found, treat the whole string as a single suggestion.
                    # This prevents incorrect splitting on abbreviations like "e.g.".
                    suggestions_list = [suggestions_str]

                if suggestions_list:
                    for suggestion in suggestions_list:
                        st.markdown(
                            f"""
                            <div style="background-color: #f0f2f6; border-left: 5px solid #3949ab; padding: 12px; margin-bottom: 10px; border-radius: 5px;">
                                <p style="margin: 0; font-size: 1.05em; color: #333;">{suggestion}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.success("✅ Great job! No critical suggestions found. Your resume seems well-aligned.")
            else:
                st.success("✅ Great job! No critical suggestions found. Your resume seems well-aligned.")
            # --- END OF IMPROVED SUGGESTIONS LOGIC ---

    if page == "🎓 Career Predictor":
        run_career_predictor()
    elif page == "📄 ATS Resume Evaluator":
        run_ats_evaluator()

else:
    # --- LOGIN/REGISTER VIEW ---
    st.title("Welcome to Future Trail 🚀")
    st.markdown("Please log in or register to continue.")
    
    try:
        login_tab, register_tab = st.tabs(["🔐 Login", "📝 Register"])

        with login_tab:
            authenticator.login(location='main')
            if st.session_state["authentication_status"] is False:
                st.error('Username/password is incorrect')
            elif st.session_state["authentication_status"] is None:
                st.warning('Please enter your username and password')

        with register_tab:
            # Show success message if registration was successful
            if st.session_state.registration_success:
                st.balloons()
                st.success('Registration successful! Please go to the "Login" tab to sign in.')
                
                st.session_state.registration_success = False
                # st.stop()

                
            
            # Custom registration form with email field
            st.subheader("📝 Register New User")
            
            with st.form("register_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    name = st.text_input("Full Name*", placeholder="Enter your full name")
                    username = st.text_input("Username*", placeholder="Choose a username")
                    email = st.text_input("Email*", placeholder="Enter your email address")
                
                with col2:
                    password = st.text_input("Password*", type="password", placeholder="Enter password")
                    confirm_password = st.text_input("Confirm Password*", type="password", placeholder="Re-enter password")
                
                submitted = st.form_submit_button("Register User", use_container_width=True)
                
                if submitted:
                    # Enhanced validation including email
                    if not all([name, username, email, password, confirm_password]):
                        st.error("❌ Please fill in all required fields")
                    elif len(password) < 6:
                        st.error("❌ Password must be at least 6 characters long")
                    elif password != confirm_password:
                        st.error("❌ Passwords do not match")
                    elif '@' not in email or '.' not in email.split('@')[-1]:
                        st.error("❌ Please enter a valid email address")
                    elif username in config['credentials']['usernames']:
                        st.error("❌ Username already exists. Please choose a different username.")
                    else:
                        try:
                            # Hash the password
                            hasher = stauth.Hasher()
                            hashed_password = hasher.hash(password)

                            # Save to database with email
                            if save_new_user(username, name, hashed_password, email):
                                # Update the in-memory credentials
                                config['credentials']['usernames'][username] = {
                                    'name': name,
                                    'password': hashed_password,
                                    'email': email
                                }

                                st.session_state.registration_success = True
                                st.rerun()
                            else:
                                st.error("❌ Registration failed. Please try again.")

                        except Exception as e:
                            st.error(f"❌ Registration error: {str(e)}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
# app.py (FINAL merged, cleaned - Option C)
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from streamlit_option_menu import option_menu
import pickle
from PIL import Image
import numpy as np
import plotly.figure_factory as ff
import seaborn as sns
import joblib
import base64
import sqlite3
import hashlib
import re
from code.DiseaseModel import DiseaseModel
from code.helper import prepare_symptoms_array
from sklearn.preprocessing import LabelEncoder

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="Disease Prediction App", layout="centered")

# ----------------------------
# DATABASE SETUP (create + upgrade)
# ----------------------------
def initialize_db():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    # create base users table if missing
    c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        email TEXT UNIQUE,
        password TEXT
    )
    """)
    # attempt to add extra columns (safe to run repeatedly)
    # profile_image stored as BLOB, plus fullname and phone
    try:
        c.execute("ALTER TABLE users ADD COLUMN fullname TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE users ADD COLUMN phone TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE users ADD COLUMN profile_image BLOB")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()

initialize_db()

# ----------------------------
# AUTHENTICATION FUNCTIONS
# ----------------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def check_login(username, password):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result and result[0] == hash_password(password):
        return True
    return False

def register_user(username, email, password):
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return "Invalid email format"
    if len(password) < 6:
        return "Password must be at least 6 characters"
    hashed = hash_password(password)
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", (username, email, hashed))
        conn.commit()
        return "Registration successful"
    except sqlite3.IntegrityError:
        return "Username or email already exists"
    finally:
        conn.close()

def get_user_details(username):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("SELECT username, email, fullname, phone, profile_image FROM users WHERE username = ?", (username,))
    row = c.fetchone()
    conn.close()
    if row:
        return {
            "username": row[0],
            "email": row[1],
            "fullname": row[2],
            "phone": row[3],
            "profile_image": row[4]
        }
    return None

def update_user_profile(username, fullname, phone, profile_image_bytes):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("""
        UPDATE users
        SET fullname = ?, phone = ?, profile_image = ?
        WHERE username = ?
    """, (fullname, phone, profile_image_bytes, username))
    conn.commit()
    conn.close()

# ----------------------------
# BACKGROUND + STYLES
# ----------------------------
def add_bg_with_white_text(image_file):
    try:
        with open(image_file, "rb") as file:
            encoded = base64.b64encode(file.read()).decode()
    except FileNotFoundError:
        # If image missing, skip background
        return
    page_bg = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        color: white !important;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.9);
        z-index: 0;
    }}
    .block-container {{
        position: relative;
        z-index: 1;
        color: white !important;
    }}
    h1, h2, h3, h4, h5, h6, p, span, div, label {{
        color: white !important;
    }}
    section[data-testid="stSidebar"] > div:first-child {{
        background-color: rgba(0, 0, 0, 0.6);
    }}
    section[data-testid="stSidebar"] * {{
        color: white !important;
    }}
    .stButton > button {{
        background-color: rgba(255, 255, 255, 0.15);
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 10px;
        padding: 0.6em 1.2em;
        font-weight: bold;
        transition: 0.3s;
        width: 100%;
    }}
    .stButton > button:hover {{
        background-color: rgba(255, 255, 255, 0.35);
        color: black !important;
        border: 1px solid white;
    }}
    .profile-round {{
        width: 44px;
        height: 44px;
        border-radius: 90%;
        object-fit: cover;
        border: 2px solid white;
    }}
    </style>
    """
    st.markdown(page_bg, unsafe_allow_html=True)

# apply background if available
add_bg_with_white_text("home.png")

# ----------------------------
# LOGIN PAGE UI
# ----------------------------
def login_page():
    st.title("Login to Disease Prediction App")
    tab1, tab2 = st.tabs(["Login", "Signup"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully")
                st.experimental_rerun()
            else:
                st.error("Invalid credentials")

    with tab2:
        username = st.text_input("Username", key="signup_username")
        email = st.text_input("Email", key="signup_email")
        password = st.text_input("Password", type="password", key="signup_password")
        if st.button("Signup"):
            result = register_user(username, email, password)
            if result == "Registration successful":
                st.success(result)
            else:
                st.error(result)

# ----------------------------
# PROFILE PAGE UI
# ----------------------------
def profile_page():
    st.title("👤 My Profile")
    user = get_user_details(st.session_state.username)
    if not user:
        st.error("Unable to load profile.")
        return

    col1, col2 = st.columns([1, 3])
    with col1:
        if user["profile_image"]:
            st.image(user["profile_image"], width=120)
        else:
            st.image(Image.new("RGB", (120,120), color=(200,200,200)), width=120)

    with col2:
        st.write(f"**Username:** {user['username']}")
        st.write(f"**Email:** {user['email']}")
        fullname = st.text_input("Full name", value=user["fullname"] or "")
        phone = st.text_input("Phone", value=user["phone"] or "")

    uploaded = st.file_uploader("Upload profile image", type=["png","jpg","jpeg"])
    profile_bytes = user["profile_image"]
    if uploaded:
        profile_bytes = uploaded.read()
        st.image(profile_bytes, width=120)

    if st.button("Save changes"):
        update_user_profile(st.session_state.username, fullname, phone, profile_bytes)
        st.success("Profile updated")
        st.experimental_rerun()

# ----------------------------
# HOME PAGE (merged prediction into home)
# ----------------------------
def home_page():
    st.markdown("<h1 style='text-align:center;'>🩺 Disease Prediction System</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; font-size:18px;'>
        Welcome to the <b>Disease Prediction Web App</b>!  
        This application uses <b>Machine Learning</b> to predict the likelihood of diseases based on your health data.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💡 Features:")
    st.markdown("- 📊 Simple and interactive input form")
    st.markdown("- 🤖 ML-based real-time predictions")
    st.markdown("- 🧠 Supports multiple diseases (Diabetes, Heart, etc.)")
    st.markdown("- 🎨 Modern design with background image and white text")
    st.info("👉 Select 'Disease Predictions' from the sidebar to start predicting!")

    # ----------------------------
# SIMPLE OFFLINE AI CHATBOT
# ----------------------------

disease_info = {
  
    "diabetes": "Diabetes is a long-term metabolic disorder where the body struggles to regulate blood sugar levels properly. This happens either because the pancreas does not produce enough insulin or because the body cannot effectively use the insulin it makes. As a result, glucose builds up in the bloodstream instead of entering the cells to provide energy. Over time, uncontrolled diabetes can damage major organs such as the heart, kidneys, eyes, and nerves. It requires consistent monitoring, medication or insulin therapy, and lifestyle changes such as a healthy diet and regular exercise.",

    "heart disease": "Heart disease refers to a group of conditions that affect the heart and its blood vessels. The most common form, coronary artery disease, occurs when blood vessels become blocked due to plaque buildup, reducing blood flow to the heart. This can lead to chest pain, heart attacks, and heart failure. Risk factors include high blood pressure, obesity, smoking, diabetes, and stress. Early diagnosis and lifestyle modifications are essential for prevention and management.",

    "parkinson": "Parkinson’s disease is a progressive neurological disorder that affects body movement. It develops when nerve cells in the brain’s substantia nigra gradually break down, causing a decrease in dopamine — a chemical responsible for smooth muscle movement. This results in tremors, muscle stiffness, slow movements, and balance difficulties. Although Parkinson’s cannot be cured, medications, physiotherapy, and lifestyle adjustments help manage the symptoms.",

    "liver disease": "Liver disease includes various conditions that interfere with the liver’s ability to function. Causes include viral infections, alcohol consumption, fatty liver, autoimmune issues, and genetic disorders. When the liver becomes inflamed or damaged, it struggles to detoxify the blood, produce proteins, and regulate metabolism. Severe liver disease can lead to cirrhosis or liver failure. Early detection and proper treatment are crucial.",

    "chronic kidney disease": "Chronic kidney disease (CKD) is the gradual loss of kidney function over time. Healthy kidneys filter waste, balance minerals, and support overall body health. When damaged due to diabetes, high blood pressure, infections, or genetic factors, their filtering ability weakens. Waste then accumulates in the body, causing complications like swelling, fatigue, and high blood pressure. Early diagnosis and lifestyle changes can slow its progression.",

    "breast cancer": "Breast cancer occurs when abnormal cells in the breast multiply uncontrollably, forming tumors that may spread to other body parts. It usually starts in the ducts or lobules and may be influenced by genetics, hormones, lifestyle, or radiation exposure. Early detection through screenings and treatments such as surgery, chemotherapy, radiation, and targeted therapy significantly improve survival chances.",

    "lung cancer": "Lung cancer is a disease where abnormal cells grow quickly in the lung tissues, forming tumors that interfere with breathing and oxygen exchange. Smoking is the leading cause, but exposure to pollutants, chemicals, or genetics can also contribute. Symptoms often appear late and include persistent cough, chest pain, and fatigue. Treatments include surgery, chemotherapy, radiation therapy, and targeted drugs.",

    "hepatitis": "Hepatitis refers to inflammation of the liver, commonly caused by viral infections like hepatitis A, B, C, D, and E. It can also result from alcohol, toxins, medications, or autoimmune conditions. The inflammation weakens the liver’s ability to process nutrients and filter blood. Symptoms include jaundice, fatigue, abdominal pain, and nausea. Some types are mild, while others become chronic and lead to serious liver damage. Vaccination and proper hygiene help prevent certain forms of hepatitis."


}

def simple_ai_bot(disease, question=""):
    disease = disease.lower().strip()

    # If disease in dictionary
    if disease in disease_info:
        base = disease_info[disease]
    else:
        base = "I don't have information about this specific disease, but it is a medical condition that may require diagnosis and treatment."

    # If follow up question asked
    if question.strip() != "":
        return base + " " + "Here is a simple answer to your question: " + question

    return base

# loading the models
diabetes_model = joblib.load("models/diabetes_model.sav")
heart_model = joblib.load("models/heart_disease_model.sav")
parkinson_model = joblib.load("models/parkinsons_model.sav")
# Load the lung cancer prediction model
lung_cancer_model = joblib.load('models/lung_cancer_model.sav')

# Load the pre-trained model
breast_cancer_model = joblib.load('models/breast_cancer.sav')

# Load the pre-trained model
chronic_disease_model = joblib.load('models/chronic_model.sav')

# Load the hepatitis prediction model
hepatitis_model = joblib.load('models/hepititisc_model.sav')


liver_model = joblib.load('models/liver_model.sav')# Load the lung cancer prediction model
lung_cancer_model = joblib.load('models/lung_cancer_model.sav')





# ----------------------------
# LOAD MODELS (unchanged)
# ----------------------------
# loading the models
diabetes_model = joblib.load("models/diabetes_model.sav")
heart_model = joblib.load("models/heart_disease_model.sav")
parkinson_model = joblib.load("models/parkinsons_model.sav")
lung_cancer_model = joblib.load('models/lung_cancer_model.sav')
breast_cancer_model = joblib.load('models/breast_cancer.sav')
chronic_disease_model = joblib.load('models/chronic_model.sav')
hepatitis_model = joblib.load('models/hepititisc_model.sav')
liver_model = joblib.load('models/liver_model.sav')
# lung_cancer_model already loaded above (kept)

# ----------------------------
# RENDER DISEASE PAGES (exact disease code grouped)
# ----------------------------
# ----------------------------
# INDIVIDUAL DISEASE PAGE FUNCTIONS
# ----------------------------

def disease_prediction_page():
    # symptom-based disease prediction (XGBoost)
    disease_model = DiseaseModel()
    disease_model.load_xgboost('model/xgboost_model.json')

    st.write('# Disease Prediction using Machine Learning')

    symptoms = st.multiselect('What are your symptoms?', options=disease_model.all_symptoms)
    X = prepare_symptoms_array(symptoms)

    if st.button('Predict', key="disease_predict"):
        prediction, prob = disease_model.predict(X)
        st.write(f'## Disease: {prediction} with {prob*100:.2f}% probability')

        tab1, tab2= st.tabs(["Description", "Precautions"])
        with tab1:
            st.write(disease_model.describe_predicted_disease())
        with tab2:
            precautions = disease_model.predicted_disease_precautions()
            for i in range(4):
                st.write(f'{i+1}. {precautions[i]}')

def diabetes_page():
    st.header("Diabetes disease prediction")
    image = Image.open('d3.jpg')
    st.image(image, caption='diabetes disease prediction')

    name = st.text_input("Name:", key="diabetes_name")
    col1, col2, col3 = st.columns(3)

    st.markdown("""
        <style>
        input[type=number]::placeholder {
            color: #999;
            opacity: 1;
        }
        </style>
    """, unsafe_allow_html=True)

    with col1:
        Pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, key="preg",
                                      help="Acceptable range: 0–20 (typically 0–17).")
        st.markdown("<small style='color:gray;'>Suggestion: 0–17</small>", unsafe_allow_html=True)

    with col2:
        Glucose = st.number_input("Glucose level", min_value=0, max_value=200, key="glu",
                                  help="Acceptable range: 0–200 mg/dL (normal fasting: 70–140).")
        st.markdown("<small style='color:gray;'>Suggestion: 70–140 mg/dL</small>", unsafe_allow_html=True)

    with col3:
        BloodPressure = st.number_input("Blood pressure value", min_value=0, max_value=150, key="bp",
                                        help="Acceptable range: 0–150 mmHg (normal: 80–120).")
        st.markdown("<small style='color:gray;'>Suggestion: 80–120 mmHg</small>", unsafe_allow_html=True)

    with col1:
        SkinThickness = st.number_input("Skin thickness value", min_value=0, max_value=100, key="skin",
                                        help="Acceptable range: 0–100 mm (typical: 10–50).")
        st.markdown("<small style='color:gray;'>Suggestion: 10–50 mm</small>", unsafe_allow_html=True)

    with col2:
        Insulin = st.number_input("Insulin value", min_value=0, max_value=900, key="insulin",
                                  help="Acceptable range: 0–900 µU/mL (typical: 15–276).")
        st.markdown("<small style='color:gray;'>Suggestion: 15–276 µU/mL</small>", unsafe_allow_html=True)

    with col3:
        BMI = st.number_input("BMI value", min_value=0.0, max_value=70.0, key="bmi",
                              help="Acceptable range: 0–70 (normal: 18.5–24.9).")
        st.markdown("<small style='color:gray;'>Suggestion: 18.5–24.9</small>", unsafe_allow_html=True)

    with col1:
        DiabetesPedigreefunction = st.number_input("Diabetes pedigree function value", min_value=0.0, max_value=3.0, key="dpf",
                                                   help="Acceptable range: 0.0–3.0 (higher = more genetic risk).")
        st.markdown("<small style='color:gray;'>Suggestion: 0.0–3.0</small>", unsafe_allow_html=True)

    with col2:
        Age = st.number_input("Age", min_value=1, max_value=120, key="age",
                              help="Acceptable range: 1–120 years (typical adult: 20–80).")
        st.markdown("<small style='color:gray;'>Suggestion: 20–80 years</small>", unsafe_allow_html=True)

    if st.button("Diabetes test result", key="diabetes_btn"):
        diabetes_prediction = diabetes_model.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreefunction, Age]]
        )
        if diabetes_prediction[0] == 1:
            diabetes_dig = "We are really sorry to say but it seems like you are Diabetic."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            diabetes_dig = "Congratulations, You are not diabetic"
            image = Image.open('negative.jpg')
            st.image(image, caption='')
        st.success(name + ' , ' + diabetes_dig)

def heart_page():
    st.header("Heart disease prediction")
    image = Image.open('heart2.jpg')
    st.image(image, caption='heart failure')

    name_h = st.text_input("Name (Heart):", key="heart_name")
    col1, col2, col3 = st.columns(3)

    with col1:
        age_h = st.number_input("Age", min_value=1, max_value=120, key="age_h",
                              help="Acceptable range: 1–120 years (typical adult: 30–80).")
        st.markdown("<small style='color:gray;'>Suggestion: 30–80 years</small>", unsafe_allow_html=True)

    with col2:
        sex = 0
        display = ("male", "female")
        options = list(range(len(display)))
        value = st.selectbox("Gender", options, format_func=lambda x: display[x], key="gender_h")
        if display[value] == "male":
            sex = 1
        elif display[value] == "female":
            sex = 0

    with col3:
        cp = 0
        display_cp = ("typical angina", "atypical angina", "non — anginal pain", "asymptotic")
        options_cp = list(range(len(display_cp)))
        value_cp = st.selectbox("Chest_Pain Type", options_cp, format_func=lambda x: display_cp[x], key="cp_h")
        if display_cp[value_cp] == "typical angina":
            cp = 0
        elif display_cp[value_cp] == "atypical angina":
            cp = 1
        elif display_cp[value_cp] == "non — anginal pain":
            cp = 2
        elif display_cp[value_cp] == "asymptotic":
            cp = 3

    with col1:
        trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=200, key="trestbps",
                                   help="Acceptable range: 50–200 mmHg (normal: 80–120).")
        st.markdown("<small style='color:gray;'>Suggestion: 80–120 mmHg</small>", unsafe_allow_html=True)

    with col2:
        chol = st.number_input("Serum Cholesterol", min_value=100, max_value=600, key="chol",
                               help="Acceptable range: 100–600 mg/dL (normal: <200).")
        st.markdown("<small style='color:gray;'>Suggestion: 150–250 mg/dL</small>", unsafe_allow_html=True)

    with col3:
        restecg = 0
        display_re = ("normal", "having ST-T wave abnormality", "left ventricular hypertrophy")
        options_re = list(range(len(display_re)))
        value_re = st.selectbox("Resting ECG", options_re, format_func=lambda x: display_re[x], key="restecg_h")
        if display_re[value_re] == "normal":
            restecg = 0
        elif display_re[value_re] == "having ST-T wave abnormality":
            restecg = 1
        elif display_re[value_re] == "left ventricular hypertrophy":
            restecg = 2

    with col1:
        thalach = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, key="thalach",
                                  help="Acceptable range: 50–250 bpm (normal: 120–200).")
        st.markdown("<small style='color:gray;'>Suggestion: 120–200 bpm</small>", unsafe_allow_html=True)

    with col2:
        oldpeak = st.number_input("ST depression induced by exercise relative to rest",
                                  min_value=0.0, max_value=10.0, key="oldpeak",
                                  help="Acceptable range: 0.0–10.0 (typical: 0.0–4.0).")
        st.markdown("<small style='color:gray;'>Suggestion: 0.0–4.0</small>", unsafe_allow_html=True)

    with col3:
        slope = 0
        display_s = ("upsloping", "flat", "downsloping")
        options_s = list(range(len(display_s)))
        value_s = st.selectbox("Peak exercise ST segment", options_s, format_func=lambda x: display_s[x], key="slope_h")
        if display_s[value_s] == "upsloping":
            slope = 0
        elif display_s[value_s] == "flat":
            slope = 1
        elif display_s[value_s] == "downsloping":
            slope = 2

    with col1:
        ca = st.number_input("Number of major vessels (0–3) colored by flourosopy",
                             min_value=0, max_value=3, key="ca",
                             help="Acceptable values: 0–3 (number of major vessels).")
        st.markdown("<small style='color:gray;'>Suggestion: 0–3</small>", unsafe_allow_html=True)

    with col2:
        thal = 0
        display_thal = ("normal", "fixed defect", "reversible defect")
        options_thal = list(range(len(display_thal)))
        value_thal = st.selectbox("Thalassemia", options_thal, format_func=lambda x: display_thal[x], key="thal_h")
        if display_thal[value_thal] == "normal":
            thal = 0
        elif display_thal[value_thal] == "fixed defect":
            thal = 1
        elif display_thal[value_thal] == "reversible defect":
            thal = 2

    with col3:
        agree = st.checkbox('Exercise induced angina', key="exang_h")
        exang = 1 if agree else 0

    with col1:
        agree1 = st.checkbox('Fasting blood sugar > 120mg/dl', key="fbs_h")
        fbs = 1 if agree1 else 0

    if st.button("Heart test result", key="heart_btn"):
        heart_prediction = heart_model.predict(
            [[age_h, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        )
        if heart_prediction[0] == 1:
            heart_dig = 'We are really sorry to say but it seems like you have Heart Disease.'
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            heart_dig = "Congratulations, You don't have Heart Disease."
            image = Image.open('negative.jpg')
            st.image(image, caption='')
        st.success(name_h + ' , ' + heart_dig)

def parkinson_page():
    st.header("Parkinson prediction")
    image = Image.open('p1.jpg')
    st.image(image, caption='parkinsons disease')

    name_p = st.text_input("Name (Parkinson):", key="parkinson_name")
    col1, col2, col3 = st.columns(3)

    with col1:
        MDVP = st.number_input("MDVP:Fo(Hz)", min_value=50.0, max_value=300.0, key="MDVP",
                               help="Range: 50–300 Hz (typical: 100–200 Hz)")
        st.markdown("<small style='color:gray;'>Suggestion: 100–200 Hz</small>", unsafe_allow_html=True)

    with col2:
        MDVPFIZ = st.number_input("MDVP:Fhi(Hz)", min_value=50.0, max_value=400.0, key="MDVPFIZ",
                                  help="Range: 50–400 Hz (typical: 150–250 Hz)")
        st.markdown("<small style='color:gray;'>Suggestion: 150–250 Hz</small>", unsafe_allow_html=True)

    with col3:
        MDVPFLO = st.number_input("MDVP:Flo(Hz)", min_value=50.0, max_value=200.0, key="MDVPFLO",
                                  help="Range: 50–200 Hz (typical: 80–150 Hz)")
        st.markdown("<small style='color:gray;'>Suggestion: 80–150 Hz</small>", unsafe_allow_html=True)

    with col1:
        MDVPJITTER = st.number_input("MDVP:Jitter(%)", min_value=0.0, max_value=1.0, key="MDVPJITTER",
                                     help="Range: 0–1% (typical: 0.0–0.01)")
        st.markdown("<small style='color:gray;'>Suggestion: 0.0–0.01</small>", unsafe_allow_html=True)

    with col2:
        MDVPJitterAbs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, max_value=0.02, key="MDVPJitterAbs",
                                        help="Range: 0–0.02 (typical: 0.0–0.005)")
        st.markdown("<small style='color:gray;'>Suggestion: 0.0–0.005</small>", unsafe_allow_html=True)

    with col3:
        MDVPRAP = st.number_input("MDVP:RAP", min_value=0.0, max_value=0.02, key="MDVPRAP",
                                  help="Range: 0–0.02 (typical: 0.0–0.006)")
        st.markdown("<small style='color:gray;'>Suggestion: 0.0–0.006</small>", unsafe_allow_html=True)

    with col2:
        MDVPPPQ = st.number_input("MDVP:PPQ", min_value=0.0, max_value=0.02, key="MDVPPPQ",
                                  help="Range: 0–0.02 (typical: 0.0–0.006)")
        st.markdown("<small style='color:gray;'>Suggestion: 0.0–0.006</small>", unsafe_allow_html=True)

    with col3:
        JitterDDP = st.number_input("Jitter:DDP", min_value=0.0, max_value=0.02, key="JitterDDP",
                                    help="Range: 0–0.02 (typical: 0.0–0.006)")
        st.markdown("<small style='color:gray;'>Suggestion: 0.0–0.006</small>", unsafe_allow_html=True)

    with col1:
        MDVPShimmer = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=0.2, key="MDVPShimmer",
                                      help="Range: 0–0.2 (typical: 0.0–0.03)")
        st.markdown("<small style='color:gray;'>Suggestion: 0.0–0.03</small>", unsafe_allow_html=True)

    with col2:
        MDVPShimmer_dB = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, max_value=2.0, key="MDVPShimmer_dB",
                                         help="Range: 0–2 dB (typical: 0–0.5)")
        st.markdown("<small style='color:gray;'>Suggestion: 0–0.5</small>", unsafe_allow_html=True)

    with col3:
        Shimmer_APQ3 = st.number_input("Shimmer:APQ3", min_value=0.0, max_value=0.2, key="Shimmer_APQ3",
                                       help="Range: 0–0.2 (typical: 0.0–0.03)")
        st.markdown("<small style='color:gray;'>Suggestion: 0.0–0.03</small>", unsafe_allow_html=True)

    with col1:
        ShimmerAPQ5 = st.number_input("Shimmer:APQ5", min_value=0.0, max_value=0.2, key="ShimmerAPQ5",
                                      help="Range: 0–0.2 (typical: 0.0–0.03)")
        st.markdown("<small style='color:gray;'>Suggestion: 0.0–0.03</small>", unsafe_allow_html=True)

    with col2:
        MDVP_APQ = st.number_input("MDVP:APQ", min_value=0.0, max_value=0.2, key="MDVP_APQ",
                                   help="Range: 0–0.2 (typical: 0.0–0.03)")
        st.markdown("<small style='color:gray;'>Suggestion: 0.0–0.03</small>", unsafe_allow_html=True)

    with col3:
        ShimmerDDA = st.number_input("Shimmer:DDA", min_value=0.0, max_value=0.5, key="ShimmerDDA",
                                     help="Range: 0–0.5 (typical: 0.0–0.05)")
        st.markdown("<small style='color:gray;'>Suggestion: 0.0–0.05</small>", unsafe_allow_html=True)

    with col1:
        NHR = st.number_input("NHR", min_value=0.0, max_value=1.0, key="NHR",
                              help="Range: 0–1 (typical: 0.0–0.3)")
        st.markdown("<small style='color:gray;'>Suggestion: 0.0–0.3</small>", unsafe_allow_html=True)

    with col2:
        HNR = st.number_input("HNR", min_value=0.0, max_value=50.0, key="HNR",
                              help="Range: 0–50 (typical: 15–30)")
        st.markdown("<small style='color:gray;'>Suggestion: 15–30</small>", unsafe_allow_html=True)

    with col2:
        RPDE = st.number_input("RPDE", min_value=0.0, max_value=2.0, key="RPDE",
                               help="Range: 0–2 (typical: 0.2–1.0)")
        st.markdown("<small style='color:gray;'>Suggestion: 0.2–1.0</small>", unsafe_allow_html=True)

    with col3:
        DFA = st.number_input("DFA", min_value=0.0, max_value=2.0, key="DFA",
                              help="Range: 0–2 (typical: 0.5–1.5)")
        st.markdown("<small style='color:gray;'>Suggestion: 0.5–1.5</small>", unsafe_allow_html=True)

    with col1:
        spread1 = st.number_input("spread1", min_value=-10.0, max_value=10.0, key="spread1",
                                  help="Range: -10–10 (typical: -2–2)")
        st.markdown("<small style='color:gray;'>Suggestion: -2–2</small>", unsafe_allow_html=True)

    with col1:
        spread2 = st.number_input("spread2", min_value=-10.0, max_value=10.0, key="spread2",
                                  help="Range: -10–10 (typical: -2–2)")
        st.markdown("<small style='color:gray;'>Suggestion: -2–2</small>", unsafe_allow_html=True)

    with col3:
        D2 = st.number_input("D2", min_value=0.0, max_value=10.0, key="D2",
                             help="Range: 0–10 (typical: 1–5)")
        st.markdown("<small style='color:gray;'>Suggestion: 1–5</small>", unsafe_allow_html=True)

    with col1:
        PPE = st.number_input("PPE", min_value=0.0, max_value=5.0, key="PPE",
                              help="Range: 0–5 (typical: 0.5–2.0)")
        st.markdown("<small style='color:gray;'>Suggestion: 0.5–2.0</small>", unsafe_allow_html=True)

    if st.button("Parkinson test result", key="parkinson_btn"):
        parkinson_prediction = parkinson_model.predict([[MDVP, MDVPFIZ, MDVPFLO, MDVPJITTER, MDVPJitterAbs, MDVPRAP, MDVPPPQ, JitterDDP,
            MDVPShimmer, MDVPShimmer_dB, Shimmer_APQ3, ShimmerAPQ5, MDVP_APQ, ShimmerDDA,
            NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
        if parkinson_prediction[0] == 1:
            parkinson_dig = 'we are really sorry to say but it seems like you have Parkinson disease'
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            parkinson_dig = "Congratulation , You don't have Parkinson disease"
            image = Image.open('negative.jpg')
            st.image(image, caption='')
        st.success(name_p + ' , ' + parkinson_dig)

def lung_cancer_page():
    # Load dataset (ensure not reloaded many times unnecessarily)
    lung_cancer_data = pd.read_csv('data/lung_cancer.csv')
    lung_cancer_data['GENDER'] = lung_cancer_data['GENDER'].map({'M': 'Male', 'F': 'Female'})

    st.header("Lung Cancer Prediction")
    image = Image.open('h.png')
    st.image(image, caption='Lung Cancer Prediction')

    name_l = st.text_input("Name (Lung):", key="lung_name")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender:", lung_cancer_data['GENDER'].unique(), key="lung_gender")
    with col2:
        age = st.number_input("Age", key="lung_age")
    with col3:
        smoking = st.selectbox("Smoking:", ['NO', 'YES'], key="smoking")
    with col1:
        yellow_fingers = st.selectbox("Yellow Fingers:", ['NO', 'YES'], key="yellow_fingers")
    with col2:
        anxiety = st.selectbox("Anxiety:", ['NO', 'YES'], key="anxiety")
    with col3:
        peer_pressure = st.selectbox("Peer Pressure:", ['NO', 'YES'], key="peer_pressure")
    with col1:
        chronic_disease = st.selectbox("Chronic Disease:", ['NO', 'YES'], key="chronic_disease")
    with col2:
        fatigue = st.selectbox("Fatigue:", ['NO', 'YES'], key="fatigue")
    with col3:
        allergy = st.selectbox("Allergy:", ['NO', 'YES'], key="allergy")
    with col1:
        wheezing = st.selectbox("Wheezing:", ['NO', 'YES'], key="wheezing")
    with col2:
        alcohol_consuming = st.selectbox("Alcohol Consuming:", ['NO', 'YES'], key="alcohol_consuming")
    with col3:
        coughing = st.selectbox("Coughing:", ['NO', 'YES'], key="coughing")
    with col1:
        shortness_of_breath = st.selectbox("Shortness of Breath:", ['NO', 'YES'], key="shortness_of_breath")
    with col2:
        swallowing_difficulty = st.selectbox("Swallowing Difficulty:", ['NO', 'YES'], key="swallowing_difficulty")
    with col3:
        chest_pain = st.selectbox("Chest Pain:", ['NO', 'YES'], key="chest_pain")

    cancer_result = ''
    if st.button("Predict Lung Cancer", key="lung_btn"):
        user_data = pd.DataFrame({
            'GENDER': [gender],
            'AGE': [age],
            'SMOKING': [smoking],
            'YELLOW_FINGERS': [yellow_fingers],
            'ANXIETY': [anxiety],
            'PEER_PRESSURE': [peer_pressure],
            'CHRONICDISEASE': [chronic_disease],
            'FATIGUE': [fatigue],
            'ALLERGY': [allergy],
            'WHEEZING': [wheezing],
            'ALCOHOLCONSUMING': [alcohol_consuming],
            'COUGHING': [coughing],
            'SHORTNESSOFBREATH': [shortness_of_breath],
            'SWALLOWINGDIFFICULTY': [swallowing_difficulty],
            'CHESTPAIN': [chest_pain]
        })
        user_data.replace({'NO': 1, 'YES': 2}, inplace=True)
        user_data.columns = user_data.columns.str.strip()
        numeric_columns = ['AGE', 'FATIGUE', 'ALLERGY', 'ALCOHOLCONSUMING', 'COUGHING', 'SHORTNESSOFBREATH']
        user_data[numeric_columns] = user_data[numeric_columns].apply(pd.to_numeric, errors='coerce')
        cancer_prediction = lung_cancer_model.predict(user_data)
        if cancer_prediction[0] == 'YES':
            cancer_result = "The model predicts that there is a risk of Lung Cancer."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            cancer_result = "The model predicts no significant risk of Lung Cancer."
            image = Image.open('negative.jpg')
            st.image(image, caption='')
        st.success(name_l + ', ' + cancer_result)

def liver_page():
    st.header("Liver Disease Prediction")
    image = Image.open('liver.jpg')
    st.image(image, caption='Liver Disease Prediction')

    name_lv = st.text_input("Name (Liver):", key="liver_name")
    col1, col2, col3 = st.columns(3)

    with col1:
        display_gender = ("Male", "Female")
        options_gender = list(range(len(display_gender)))
        value_gender = st.selectbox("Gender", options_gender, format_func=lambda x: display_gender[x], key="l_gender")
        Sex = 0 if display_gender[value_gender] == "Male" else 1

    with col2:
        age_lv = st.number_input("Enter your age", min_value=1, max_value=120, key="age_lv")
        st.markdown("<small style='color:gray;'>Suggestion (normal range): 20–80 years</small>", unsafe_allow_html=True)

    with col3:
        Total_Bilirubin = st.number_input("Total Bilirubin (mg/dL)", min_value=0.0, max_value=50.0, step=0.1, key="tb")
        st.markdown("<small style='color:gray;'>Suggestion (normal range): 0.1–1.2 mg/dL</small>", unsafe_allow_html=True)

    with col1:
        Direct_Bilirubin = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, max_value=50.0, step=0.1, key="db")
        st.markdown("<small style='color:gray;'>Suggestion (normal range): 0.0–0.3 mg/dL</small>", unsafe_allow_html=True)

    with col2:
        Alkaline_Phosphotase = st.number_input("Alkaline Phosphatase (U/L)", min_value=0, max_value=1000, key="alp")
        st.markdown("<small style='color:gray;'>Suggestion (normal range): 44–147 U/L</small>", unsafe_allow_html=True)

    with col3:
        Alamine_Aminotransferase = st.number_input("Alamine Aminotransferase (U/L)", min_value=0, max_value=1000, key="alt")
        st.markdown("<small style='color:gray;'>Suggestion (normal range): 7–56 U/L</small>", unsafe_allow_html=True)

    with col1:
        Aspartate_Aminotransferase = st.number_input("Aspartate Aminotransferase (U/L)", min_value=0, max_value=1000, key="ast")
        st.markdown("<small style='color:gray;'>Suggestion (normal range): 10–40 U/L</small>", unsafe_allow_html=True)

    with col2:
        Total_Protiens = st.number_input("Total Proteins (g/dL)", min_value=0.0, max_value=20.0, step=0.1, key="tp")
        st.markdown("<small style='color:gray;'>Suggestion (normal range): 6–8 g/dL</small>", unsafe_allow_html=True)

    with col3:
        Albumin = st.number_input("Albumin (g/dL)", min_value=0.0, max_value=10.0, step=0.1, key="albumin")
        st.markdown("<small style='color:gray;'>Suggestion (normal range): 3.5–5.0 g/dL</small>", unsafe_allow_html=True)

    with col1:
        Albumin_and_Globulin_Ratio = st.number_input("Albumin/Globulin Ratio", min_value=0.0, max_value=5.0, step=0.01, key="agr")
        st.markdown("<small style='color:gray;'>Suggestion (normal range): 1.0–2.5</small>", unsafe_allow_html=True)

    if st.button("Liver Test Result", key="liver_btn"):
        liver_prediction = liver_model.predict([[Sex, age_lv, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio]])
        if liver_prediction[0] == 1:
            liver_dig = "We are really sorry to say, but it seems like you have liver disease."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            liver_dig = "Congratulations, you don't have liver disease."
            image = Image.open('negative.jpg')
            st.image(image, caption='')
        st.success(name_lv + ', ' + liver_dig)

def hepatitis_page():
    st.header("Hepatitis Prediction")
    image = Image.open('h.png')
    st.image(image, caption='Hepatitis Prediction')

    name_hep = st.text_input("Name (Hepatitis):", key="hep_name")
    col1, col2, col3 = st.columns(3)

    with col1:
        age_hep = st.number_input("Enter your age", min_value=1, max_value=120, help="Age in years", key="age_hep")
        st.markdown("<small style='color:gray;'>Suggestion: 20–80 years</small>", unsafe_allow_html=True)

    with col2:
        sex_hep = st.selectbox("Gender", ["Male", "Female"], key="sex_hep")
        sex_hep = 1 if sex_hep == "Male" else 2

    with col3:
        total_bilirubin = st.number_input("Total Bilirubin (mg/dL)", min_value=0.0, step=0.1, key="tb_hep")

    with col1:
        direct_bilirubin = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, step=0.1, key="db_hep")

    with col2:
        alkaline_phosphatase = st.number_input("Alkaline Phosphatase (U/L)", min_value=0, key="alp_hep")

    with col3:
        alamine_aminotransferase = st.number_input("Alamine Aminotransferase (U/L)", min_value=0, key="alt_hep")

    with col1:
        aspartate_aminotransferase = st.number_input("Aspartate Aminotransferase (U/L)", min_value=0, key="ast_hep")

    with col2:
        total_proteins = st.number_input("Total Proteins (g/dL)", min_value=0.0, step=0.1, key="tp_hep")

    with col3:
        albumin = st.number_input("Albumin (g/dL)", min_value=0.0, step=0.1, key="albumin_hep")

    with col1:
        albumin_and_globulin_ratio = st.number_input("Albumin/Globulin Ratio", min_value=0.0, step=0.01, key="agr_hep")

    with col2:
        ggt_value = st.number_input("GGT (U/L)", min_value=0, key="ggt_hep")

    with col3:
        prot_value = st.number_input("PROT (g/dL)", min_value=0.0, step=0.1, key="prot_hep")

    if st.button("Predict Hepatitis", key="hep_btn"):
        user_data = pd.DataFrame({
            'Age': [age_hep],
            'Sex': [sex_hep],
            'ALB': [total_bilirubin],
            'ALP': [direct_bilirubin],
            'ALT': [alkaline_phosphatase],
            'AST': [alamine_aminotransferase],
            'BIL': [aspartate_aminotransferase],
            'CHE': [total_proteins],
            'CHOL': [albumin],
            'CREA': [albumin_and_globulin_ratio],
            'GGT': [ggt_value],
            'PROT': [prot_value]
        })
        hepatitis_prediction = hepatitis_model.predict(user_data)
        if hepatitis_prediction[0] == 1:
            hepatitis_result = "We are really sorry to say but it seems like you have Hepatitis."
            image = Image.open('positive.jpg')
            st.image(image, caption='')
        else:
            hepatitis_result = 'Congratulations, you do not have Hepatitis.'
            image = Image.open('negative.jpg')
            st.image(image, caption='')
        st.success(name_hep + ', ' + hepatitis_result)

def jaundice_page():
    st.header("Jaundice disease prediction")
    image = Image.open('j.jpg')
    st.image(image, caption='Jaundice disease prediction')

    name_j = st.text_input("Name (Jaundice):", key="jaundice_name")
    col1, col2, col3 = st.columns(3)

    with col1:
        age_j = st.number_input("Enter your age", key="age_j") # 2
    with col2:
        Sex_j = 0
        display_j = ("male", "female")
        options_j = list(range(len(display_j)))
        value_j = st.selectbox("Gender", options_j, format_func=lambda x: display_j[x], key="sex_j")
        if display_j[value_j] == "male":
            Sex_j = 0
        elif display_j[value_j] == "female":
            Sex_j = 1
    with col3:
        Total_Bilirubin_j = st.number_input("Enter your Total_Bilirubin", key="tb_j") # 3
    with col1:
        Direct_Bilirubin_j = st.number_input("Enter your Direct_Bilirubin", key="db_j")# 4
    with col2:
        Alkaline_Phosphotase_j = st.number_input("Enter your Alkaline_Phosphotase", key="alp_j") # 5
    with col3:
        Alamine_Aminotransferase_j = st.number_input("Enter your Alamine_Aminotransferase", key="alt_j") # 6
    with col1:
        Total_Protiens_j = st.number_input("Enter your Total_Protiens", key="tp_j")# 8
    with col2:
        Albumin_j = st.number_input("Enter your Albumin", key="albumin_j") # 9

    if st.button("Jaundice test result", key="jaundice_btn"):
        try:
            jaundice_prediction = jaundice_model.predict([[age_j, Sex_j, Total_Bilirubin_j, Direct_Bilirubin_j, Alkaline_Phosphotase_j, Alamine_Aminotransferase_j, Total_Protiens_j, Albumin_j]])
            if jaundice_prediction[0] == 1:
                image = Image.open('positive.jpg')
                st.image(image, caption='')
                jaundice_dig = "We are really sorry to say but it seems like you have Jaundice."
            else:
                image = Image.open('negative.jpg')
                st.image(image, caption='')
                jaundice_dig = "Congratulations, You don't have Jaundice."
            st.success(name_j + ' , ' + jaundice_dig)
        except Exception as e:
            st.error("Jaundice model not available or an error occurred: " + str(e))

def chronic_kidney_page():
    st.header("Chronic Kidney Disease Prediction")
    name_k = st.text_input("Name (Kidney):", key="kidney_name")
    col1, col2, col3 = st.columns(3)

    with col1:
        age_k = st.slider("Enter your age", 1, 100, 25, key="age_k")  # 2
    with col2:
        bp = st.slider("Enter your Blood Pressure", 50, 200, 120, key="bp_k")  # Add your own ranges
    with col3:
        sg = st.slider("Enter your Specific Gravity", 1.0, 1.05, 1.02, key="sg_k")  # Add your own ranges

    with col1:
        al = st.slider("Enter your Albumin", 0, 5, 0, key="al_k")  # Add your own ranges
    with col2:
        su = st.slider("Enter your Sugar", 0, 5, 0, key="su_k")  # Add your own ranges
    with col3:
        rbc = st.selectbox("Red Blood Cells", ["Normal", "Abnormal"], key="rbc_k")
        rbc = 1 if rbc == "Normal" else 0

    with col1:
        pc = st.selectbox("Pus Cells", ["Normal", "Abnormal"], key="pc_k")
        pc = 1 if pc == "Normal" else 0
    with col2:
        pcc = st.selectbox("Pus Cell Clumps", ["Present", "Not Present"], key="pcc_k")
        pcc = 1 if pcc == "Present" else 0
    with col3:
        ba = st.selectbox("Bacteria", ["Present", "Not Present"], key="ba_k")
        ba = 1 if ba == "Present" else 0

    with col1:
        bgr = st.slider("Enter your Blood Glucose Random", 50, 200, 120, key="bgr_k")  # Add your own ranges
    with col2:
        bu = st.slider("Enter your Blood Urea", 10, 200, 60, key="bu_k")  # Add your own ranges
    with col3:
        sc = st.slider("Enter your Serum Creatinine", 0, 10, 3, key="sc_k")  # Add your own ranges

    with col1:
        sod = st.slider("Enter your Sodium", 100, 200, 140, key="sod_k")  # Add your own ranges
    with col2:
        pot = st.slider("Enter your Potassium", 2, 7, 4, key="pot_k")  # Add your own ranges
    with col3:
        hemo = st.slider("Enter your Hemoglobin", 3, 17, 12, key="hemo_k")  # Add your own ranges

    with col1:
        pcv = st.slider("Enter your Packed Cell Volume", 20, 60, 40, key="pcv_k")  # Add your own ranges
    with col2:
        wc = st.slider("Enter your White Blood Cell Count", 2000, 20000, 10000, key="wc_k")  # Add your own ranges
    with col3:
        rc = st.slider("Enter your Red Blood Cell Count", 2, 8, 4, key="rc_k")  # Add your own ranges

    with col1:
        htn = st.selectbox("Hypertension", ["Yes", "No"], key="htn_k")
        htn = 1 if htn == "Yes" else 0
    with col2:
        dm = st.selectbox("Diabetes Mellitus", ["Yes", "No"], key="dm_k")
        dm = 1 if dm == "Yes" else 0
    with col3:
        cad = st.selectbox("Coronary Artery Disease", ["Yes", "No"], key="cad_k")
        cad = 1 if cad == "Yes" else 0

    with col1:
        appet = st.selectbox("Appetite", ["Good", "Poor"], key="appet_k")
        appet = 1 if appet == "Good" else 0
    with col2:
        pe = st.selectbox("Pedal Edema", ["Yes", "No"], key="pe_k")
        pe = 1 if pe == "Yes" else 0
    with col3:
        ane = st.selectbox("Anemia", ["Yes", "No"], key="ane_k")
        ane = 1 if ane == "Yes" else 0

    if st.button("Predict Chronic Kidney Disease", key="kidney_btn"):
        user_input = pd.DataFrame({
            'age': [age_k],
            'bp': [bp],
            'sg': [sg],
            'al': [al],
            'su': [su],
            'rbc': [rbc],
            'pc': [pc],
            'pcc': [pcc],
            'ba': [ba],
            'bgr': [bgr],
            'bu': [bu],
            'sc': [sc],
            'sod': [sod],
            'pot': [pot],
            'hemo': [hemo],
            'pcv': [pcv],
            'wc': [wc],
            'rc': [rc],
            'htn': [htn],
            'dm': [dm],
            'cad': [cad],
            'appet': [appet],
            'pe': [pe],
            'ane': [ane]
        })
        kidney_prediction = chronic_disease_model.predict(user_input)
        if kidney_prediction[0] == 1:
            image = Image.open('positive.jpg')
            st.image(image, caption='')
            kidney_prediction_dig = "we are really sorry to say but it seems like you have kidney disease."
        else:
            image = Image.open('negative.jpg')
            st.image(image, caption='')
            kidney_prediction_dig = "Congratulations, You don't have kidney disease."
        st.success(name_k + ' , ' + kidney_prediction_dig)

def breast_cancer_page():
    st.header("Breast Cancer Prediction")
    name_b = st.text_input("Name (Breast):", key="breast_name")
    col1, col2, col3 = st.columns(3)

    with col1:
        radius_mean = st.slider("Enter your Radius Mean", 6.0, 30.0, 15.0, key="radius_mean")
        texture_mean = st.slider("Enter your Texture Mean", 9.0, 40.0, 20.0, key="texture_mean")
        perimeter_mean = st.slider("Enter your Perimeter Mean", 43.0, 190.0, 90.0, key="perimeter_mean")

    with col2:
        area_mean = st.slider("Enter your Area Mean", 143.0, 2501.0, 750.0, key="area_mean")
        smoothness_mean = st.slider("Enter your Smoothness Mean", 0.05, 0.25, 0.1, key="smoothness_mean")
        compactness_mean = st.slider("Enter your Compactness Mean", 0.02, 0.3, 0.15, key="compactness_mean")

    with col3:
        concavity_mean = st.slider("Enter your Concavity Mean", 0.0, 0.5, 0.2, key="concavity_mean")
        concave_points_mean = st.slider("Enter your Concave Points Mean", 0.0, 0.2, 0.1, key="concave_points_mean")
        symmetry_mean = st.slider("Enter your Symmetry Mean", 0.1, 1.0, 0.5, key="symmetry_mean")

    with col1:
        fractal_dimension_mean = st.slider("Enter your Fractal Dimension Mean", 0.01, 0.1, 0.05, key="fractal_dimension_mean")
        radius_se = st.slider("Enter your Radius SE", 0.1, 3.0, 1.0, key="radius_se")
        texture_se = st.slider("Enter your Texture SE", 0.2, 2.0, 1.0, key="texture_se")

    with col2:
        perimeter_se = st.slider("Enter your Perimeter SE", 1.0, 30.0, 10.0, key="perimeter_se")
        area_se = st.slider("Enter your Area SE", 6.0, 500.0, 150.0, key="area_se")
        smoothness_se = st.slider("Enter your Smoothness SE", 0.001, 0.03, 0.01, key="smoothness_se")

    with col3:
        compactness_se = st.slider("Enter your Compactness SE", 0.002, 0.2, 0.1, key="compactness_se")
        concavity_se = st.slider("Enter your Concavity SE", 0.0, 0.05, 0.02, key="concavity_se")
        concave_points_se = st.slider("Enter your Concave Points SE", 0.0, 0.03, 0.01, key="concave_points_se")

    with col1:
        symmetry_se = st.slider("Enter your Symmetry SE", 0.1, 1.0, 0.5, key="symmetry_se")
        fractal_dimension_se = st.slider("Enter your Fractal Dimension SE", 0.01, 0.1, 0.05, key="fractal_dimension_se")

    with col2:
        radius_worst = st.slider("Enter your Radius Worst", 7.0, 40.0, 20.0, key="radius_worst")
        texture_worst = st.slider("Enter your Texture Worst", 12.0, 50.0, 25.0, key="texture_worst")
        perimeter_worst = st.slider("Enter your Perimeter Worst", 50.0, 250.0, 120.0, key="perimeter_worst")

    with col3:
        area_worst = st.slider("Enter your Area Worst", 185.0, 4250.0, 1500.0, key="area_worst")
        smoothness_worst = st.slider("Enter your Smoothness Worst", 0.07, 0.3, 0.15, key="smoothness_worst")
        compactness_worst = st.slider("Enter your Compactness Worst", 0.03, 0.6, 0.3, key="compactness_worst")

    with col1:
        concavity_worst = st.slider("Enter your Concavity Worst", 0.0, 0.8, 0.4, key="concavity_worst")
        concave_points_worst = st.slider("Enter your Concave Points Worst", 0.0, 0.2, 0.1, key="concave_points_worst")
        symmetry_worst = st.slider("Enter your Symmetry Worst", 0.1, 1.0, 0.5, key="symmetry_worst")
    with col2:
        fractal_dimension_worst = st.slider("Enter your fractal dimension worst",0.01,0.2,0.1)
      
    if st.button("Predict Breast Cancer", key="breast_btn"):
        user_input = pd.DataFrame({
            'radius_mean': [radius_mean],
            'texture_mean': [texture_mean],
            'perimeter_mean': [perimeter_mean],
            'area_mean': [area_mean],
            'smoothness_mean': [smoothness_mean],
            'compactness_mean': [compactness_mean],
            'concavity_mean': [concavity_mean],
            'concave points_mean': [concave_points_mean],
            'symmetry_mean': [symmetry_mean],
            'fractal_dimension_mean': [fractal_dimension_mean],
            'radius_se': [radius_se],
            'texture_se': [texture_se],
            'perimeter_se': [perimeter_se],
            'area_se': [area_se],
            'smoothness_se': [smoothness_se],
            'compactness_se': [compactness_se],
            'concavity_se': [concavity_se],
            'concave points_se': [concave_points_se],
            'symmetry_se': [symmetry_se],
            'fractal_dimension_se': [fractal_dimension_se],
            'radius_worst': [radius_worst],
            'texture_worst': [texture_worst],
            'perimeter_worst': [perimeter_worst],
            'area_worst': [area_worst],
            'smoothness_worst': [smoothness_worst],
            'compactness_worst': [compactness_worst],
            'concavity_worst': [concavity_worst],
            'concave points_worst': [concave_points_worst],
            'symmetry_worst': [symmetry_worst],
            'fractal_dimension_worst': [fractal_dimension_worst],
        })
        breast_cancer_prediction = breast_cancer_model.predict(user_input)
        if breast_cancer_prediction[0] == 1:
            image = Image.open('positive.jpg')
            st.image(image, caption='')
            breast_cancer_result = "The model predicts that you have Breast Cancer."
        else:
            image = Image.open('negative.jpg')
            st.image(image, caption='')
            breast_cancer_result = "The model predicts that you don't have Breast Cancer."
        st.success(breast_cancer_result)

# Dispatcher: call the right page based on the option menu selection
def render_disease_pages(selected):
    if selected == "Disease Prediction":
        disease_prediction_page()
    elif selected == "Diabetes Prediction":
        diabetes_page()
    elif selected == "Heart disease Prediction":
        heart_page()
    elif selected == "Parkison Prediction":
        parkinson_page()
    elif selected == "Liver prediction":
        liver_page()
    elif selected == "Hepatitis prediction":
        hepatitis_page()
    elif selected == "Lung Cancer Prediction":
        lung_cancer_page()
    elif selected == "Chronic Kidney prediction":
        chronic_kidney_page()
    elif selected == "Breast Cancer Prediction":
        breast_cancer_page()


    # --- End original disease pages ---

# ----------------------------
# MAIN APPLICATION
# ----------------------------
def main():
    # session defaults
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None

    # If not logged in, show login UI and stop
    if not st.session_state.logged_in:
        login_page()
        return

    # Logged in: show sidebar navigation
    st.sidebar.title("🔍 Navigation")

    # show small profile photo (if available)
    user = get_user_details(st.session_state.username)
    if user and user.get("profile_image"):
        try:
            img_data = user["profile_image"]
            st.sidebar.image(img_data, width=44)
            # st.sidebar.title("profilename")
        except Exception:
            pass

    # main choices
    choice = st.sidebar.radio("Go to:", ["Home", "Disease Predictions", "Profile", "Logout"])

    # ---------------- PAGE ROUTING INSIDE MAIN() ----------------
    if choice == "Home":
        home_page()

    elif choice == "Disease Predictions":

        with st.sidebar:
            selected = option_menu(
                'Multiple Disease Prediction',
                [
                    'Disease Prediction',
                    'Diabetes Prediction',
                    'Heart disease Prediction',
                    'Parkison Prediction',
                    'Liver prediction',
                    'Hepatitis prediction',
                    'Lung Cancer Prediction',
                    'Chronic Kidney prediction',
                    'Breast Cancer Prediction',
                ]
            )

        render_disease_pages(selected)# ----------------------------
    # SIMPLE AI CHATBOT IN SIDEBAR
    # ----------------------------
    st.markdown("---")
    st.subheader("🧠 Simple AI Health Chatbot")

    disease_query = st.text_input("Disease name:", key="ai_disease")
    followup_query = st.text_input("Ask a question (optional):", key="ai_followup")

    if st.button("Ask AI", key="ai_button"):
        if disease_query.strip() == "":
            st.warning("Please enter a disease name.")
        else:
            st.write("### AI Response")
            st.write(simple_ai_bot(disease_query, followup_query))


        


    elif choice == "Profile":
        profile_page()

    elif choice == "Logout":
        st.session_state.logged_in = False
        st.session_state.username = None
        st.experimental_rerun()


# ---------------- RUN MAIN APP ----------------
if __name__ == "__main__":
    main()

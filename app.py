import streamlit as st
import joblib
import numpy as np
import pandas as pd
import sqlite3
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ======================
# Config
# ======================
st.set_page_config(page_title="Diabetes App", layout="wide")

# ======================
# Load model & scaler
# ======================
model = joblib.load("diabetes_best_model.pkl")
scaler = joblib.load("scaler.pkl")

# ======================
# Database
# ======================
conn = sqlite3.connect("database.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pregnancies INTEGER,
    glucose REAL,
    bmi REAL,
    prediction INTEGER,
    probability REAL
)
""")
conn.commit()

# ======================
# Sidebar
# ======================
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["User Prediction", "Admin Dashboard"])

# ======================
# USER PAGE
# ======================
if page == "User Prediction":

    st.title("🩺 Diabetes Prediction System")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, value=2)
        glucose = st.number_input("Glucose", 0, 300, value=120)
        bp = st.number_input("Blood Pressure", 0, 200, value=70)
        skin = st.number_input("Skin Thickness", 0, 100, value=20)

    with col2:
        insulin = st.number_input("Insulin", 0, 900, value=80)
        bmi = st.number_input("BMI", 0.0, 70.0, value=25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, value=0.5)
        age = st.number_input("Age", 1, 120, value=30)

    if st.button("🔍 Predict"):

        features = [
            pregnancies, glucose, bp, skin,
            insulin, bmi, dpf, age
        ]

        data = np.array(features).reshape(1, -1)
        data_scaled = scaler.transform(data)

        prediction = model.predict(data_scaled)[0]
        prob = model.predict_proba(data_scaled)[0][1] * 100

        # 🎯 نتيجة بشكل احترافي
        st.markdown("## 🧾 Result")

        if prediction == 1:
            st.markdown(f"""
            <div style='padding:20px;background-color:#ff4d4d;border-radius:10px;color:white'>
            <h2>⚠️ Diabetic</h2>
            <h3>Risk: {prob:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='padding:20px;background-color:#28a745;border-radius:10px;color:white'>
            <h2>✅ Not Diabetic</h2>
            <h3>Confidence: {100 - prob:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)

        # Save to DB
        cursor.execute("""
        INSERT INTO predictions (pregnancies, glucose, bmi, prediction, probability)
        VALUES (?, ?, ?, ?, ?)
        """, (pregnancies, glucose, bmi, int(prediction), float(prob)))

        conn.commit()

# ======================
# ADMIN PAGE
# ======================
elif page == "Admin Dashboard":

    st.title("🛠️ Admin Dashboard")

    tab1, tab2, tab3 = st.tabs(["📋 Logs", "📊 Metrics", "🔄 Retrain"])

    # ======================
    # Logs
    # ======================
    with tab1:
        st.subheader("Prediction Logs")

        cursor.execute("SELECT * FROM predictions")
        rows = cursor.fetchall()

        if rows:
            cursor.execute("PRAGMA table_info(predictions)")
            cols = [col[1] for col in cursor.fetchall()]

            df = pd.DataFrame(rows, columns=cols)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("No data yet")

    # ======================
    # Metrics
    # ======================
    with tab2:
        st.subheader("Model Performance")

        try:
            df = pd.read_csv("diabetes.csv")

            X = df.drop("Outcome", axis=1)
            y = df["Outcome"]

            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Accuracy", f"{accuracy_score(y, y_pred):.2f}")
            c2.metric("Precision", f"{precision_score(y, y_pred):.2f}")
            c3.metric("Recall", f"{recall_score(y, y_pred):.2f}")
            c4.metric("F1 Score", f"{f1_score(y, y_pred):.2f}")

        except:
            st.error("Dataset not found!")

    # ======================
    # Retrain
    # ======================
    with tab3:
        st.subheader("Retrain Model")

        file = st.file_uploader("Upload Dataset (CSV)")

        if file:
            df = pd.read_csv(file)

            if st.button("🚀 Train Model"):
                from sklearn.ensemble import RandomForestClassifier

                X = df.drop("Outcome", axis=1)
                y = df["Outcome"]

                X_scaled = scaler.fit_transform(X)

                model = RandomForestClassifier()
                model.fit(X_scaled, y)

                joblib.dump(model, "diabetes_best_model.pkl")

                st.success("Model retrained successfully!")
# File name: app.py

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("logistic_model.pkl")
features = joblib.load("model_features.pkl")

st.title("🚢 Titanic Survival Prediction")
st.markdown("Fill in the details below to predict if a passenger would survive.")

with st.form("prediction_form"):
    Pclass = st.selectbox("Passenger Class", [1, 2, 3])
    Sex = st.selectbox("Sex", ["male", "female"])
    Age = st.slider("Age", 0, 80, 30)
    SibSp = st.number_input("No. of Siblings/Spouses Aboard", 0, 8, 0)
    Parch = st.number_input("No. of Parents/Children Aboard", 0, 6, 0)
    Fare = st.slider("Fare Paid", 0.0, 500.0, 32.2)
    Embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"])
    submitted = st.form_submit_button("Predict")

if submitted:
    Sex = 0 if Sex == 'male' else 1
    Embarked_C = 1 if Embarked == 'C' else 0
    Embarked_Q = 1 if Embarked == 'Q' else 0

    input_data = pd.DataFrame([{
        'Pclass': Pclass,
        'Sex': Sex,
        'Age': Age,
        'SibSp': SibSp,
        'Parch': Parch,
        'Fare': Fare,
        'Embarked_C': Embarked_C,
        'Embarked_Q': Embarked_Q
    }])

    input_data = input_data.reindex(columns=features, fill_value=0)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Prediction: Survived ({probability:.2f} probability)")
    else:
        st.error(f"Prediction: Did Not Survive ({probability:.2f} probability)")

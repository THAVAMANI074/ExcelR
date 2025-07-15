{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ab1a8874-96a3-4b7b-94ce-c443d7b31bf9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# File name: app.py\n",
    "\n",
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import joblib\n",
    "\n",
    "model = joblib.load(\"logistic_model.pkl\")\n",
    "features = joblib.load(\"model_features.pkl\")\n",
    "\n",
    "st.title(\"🚢 Titanic Survival Prediction\")\n",
    "st.markdown(\"Fill in the details below to predict if a passenger would survive.\")\n",
    "\n",
    "with st.form(\"prediction_form\"):\n",
    "    Pclass = st.selectbox(\"Passenger Class\", [1, 2, 3])\n",
    "    Sex = st.selectbox(\"Sex\", [\"male\", \"female\"])\n",
    "    Age = st.slider(\"Age\", 0, 80, 30)\n",
    "    SibSp = st.number_input(\"No. of Siblings/Spouses Aboard\", 0, 8, 0)\n",
    "    Parch = st.number_input(\"No. of Parents/Children Aboard\", 0, 6, 0)\n",
    "    Fare = st.slider(\"Fare Paid\", 0.0, 500.0, 32.2)\n",
    "    Embarked = st.selectbox(\"Port of Embarkation\", [\"S\", \"C\", \"Q\"])\n",
    "    submitted = st.form_submit_button(\"Predict\")\n",
    "\n",
    "if submitted:\n",
    "    Sex = 0 if Sex == 'male' else 1\n",
    "    Embarked_C = 1 if Embarked == 'C' else 0\n",
    "    Embarked_Q = 1 if Embarked == 'Q' else 0\n",
    "\n",
    "    input_data = pd.DataFrame([{\n",
    "        'Pclass': Pclass,\n",
    "        'Sex': Sex,\n",
    "        'Age': Age,\n",
    "        'SibSp': SibSp,\n",
    "        'Parch': Parch,\n",
    "        'Fare': Fare,\n",
    "        'Embarked_C': Embarked_C,\n",
    "        'Embarked_Q': Embarked_Q\n",
    "    }])\n",
    "\n",
    "    input_data = input_data.reindex(columns=features, fill_value=0)\n",
    "\n",
    "    prediction = model.predict(input_data)[0]\n",
    "    probability = model.predict_proba(input_data)[0][1]\n",
    "\n",
    "    if prediction == 1:\n",
    "        st.success(f\"Prediction: Survived ({probability:.2f} probability)\")\n",
    "    else:\n",
    "        st.error(f\"Prediction: Did Not Survive ({probability:.2f} probability)\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

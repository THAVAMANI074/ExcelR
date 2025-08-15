import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os

# === Load model ===
MODEL_PATH = os.path.join("models", "best_price_classifier_xgb_10bins.joblib")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# === Features ===
FEATURES = [
    'kilometerdriven', 'ownernumber', 'fueltype_num', 'transmission_num',
    'isc24assured_num', 'age', 'km_per_age', 'price_per_km', 'log_km',
    'log_age', 'age_squared', 'km_age_interaction'
]

# Price bins for mapping predicted class to price range
price_bins = [
    (0, 50000), (50001, 150000), (150001, 250000), (250001, 350000),
    (350001, 450000), (450001, 600000), (600001, 900000), (900001, 1200000),
    (1200001, 1800000), (1800001, 4000000)
]

st.title("🚗 Car Price Prediction (5-Column Input)")

uploaded_file = st.file_uploader(
    "Upload CSV with columns: kilometerdriven, ownernumber, fueltype_num, transmission_num, isc24assured_num",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)

        # Check required columns
        required_cols = ['kilometerdriven', 'ownernumber', 'fueltype_num', 'transmission_num', 'isc24assured_num']
        missing = [c for c in required_cols if c not in df_input.columns]
        if missing:
            st.error(f"❌ Missing required columns: {missing}")
        else:
            # Calculate car age
            if 'year' in df_input.columns:
                current_year = datetime.now().year
                df_input['age'] = current_year - df_input['year']
            else:
                default_age = st.number_input("Default Car Age (years)", min_value=0.0, step=0.5, value=5.0)
                df_input['age'] = default_age

            # Feature engineering
            df_input['km_per_age'] = df_input['kilometerdriven'] / df_input['age'].replace(0, 1)
            df_input['price_per_km'] = 0
            df_input['log_km'] = np.log1p(df_input['kilometerdriven'])
            df_input['log_age'] = np.log1p(df_input['age'] + 1)
            df_input['age_squared'] = df_input['age'] ** 2
            df_input['km_age_interaction'] = df_input['kilometerdriven'] * df_input['age']

            # Prediction
            df_model_input = df_input[FEATURES]
            pred_classes = model.predict(df_model_input)
            pred_probs = model.predict_proba(df_model_input)

            # Add predictions
            df_input['Predicted_Bin'] = pred_classes
            df_input['Predicted_Range'] = [
                f"₹{low:,} - ₹{high:,}" for low, high in [price_bins[c] for c in pred_classes]
            ]
            df_input['Estimated_Price'] = [
                sum(prob * ((low + high) / 2) for prob, (low, high) in zip(probs, price_bins))
                for probs in pred_probs
            ]

            # Show results
            st.subheader("📊 Prediction Results")
            st.dataframe(df_input)

            # Download button
            csv_out = df_input.to_csv(index=False)
            st.download_button(
                "📥 Download Predictions CSV",
                data=csv_out,
                file_name="predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")

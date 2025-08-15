from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load(r'C:\Users\Thavamani\Desktop\ExcelR-Project1\best_price_classifier_xgb_10bins.joblib')

FEATURES = [
    'kilometerdriven', 'ownernumber', 'fueltype_num', 'transmission_num',
    'isc24assured_num', 'age', 'km_per_age', 'price_per_km', 'log_km',
    'log_age', 'age_squared', 'km_age_interaction'
]

price_bins = [
    (0, 50000), (50001, 150000), (150001, 250000), (250001, 350000),
    (350001, 450000), (450001, 600000), (600001, 900000), (900001, 1200000),
    (1200001, 1800000), (1800001, 4000000)
]

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            kilometerdriven = float(request.form['kilometerdriven'])
            ownernumber = int(request.form['ownernumber'])
            fueltype_num = int(request.form['fueltype_num'])
            transmission_num = int(request.form['transmission_num'])
            isc24assured_num = int(request.form['isc24assured_num'])
            age = float(request.form['age'])

            df_input = pd.DataFrame([{
                'kilometerdriven': kilometerdriven,
                'ownernumber': ownernumber,
                'fueltype_num': fueltype_num,
                'transmission_num': transmission_num,
                'isc24assured_num': isc24assured_num,
                'age': age
            }])

            df_input['km_per_age'] = df_input['kilometerdriven'] / df_input['age'].replace(0, 1)
            df_input['price_per_km'] = 0
            df_input['log_km'] = np.log1p(df_input['kilometerdriven'])
            df_input['log_age'] = np.log1p(df_input['age'] + 1)
            df_input['age_squared'] = df_input['age'] ** 2
            df_input['km_age_interaction'] = df_input['kilometerdriven'] * df_input['age']

            df_input = df_input[FEATURES]

            pred_class = model.predict(df_input)[0]
            pred_probs = model.predict_proba(df_input)[0]
            predicted_range = price_bins[pred_class]
            estimated_price = sum(prob * ((low + high) / 2) for prob, (low, high) in zip(pred_probs, price_bins))

            return render_template('index.html',
                                   predicted_bin=pred_class,
                                   predicted_range=f"₹{predicted_range[0]:,} to ₹{predicted_range[1]:,}",
                                   estimated_price=f"₹{estimated_price:,.2f}",
                                   probabilities=[f"{p:.3f}" for p in pred_probs])

        except Exception as e:
            return render_template('index.html', error=str(e))

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

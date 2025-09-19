from flask import Flask,render_template,request
import joblib
import numpy as np


app = Flask(__name__)

model_path = "artifacts/model/model.pkl"
scaler_path = "artifacts/processed/scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/')
def home():
    return render_template("index.html" , prediction=None)

@app.route('/predict',methods=["POST"])
def predict():
    try:
        healthcare_cost = float(request.form["healthcare_costs"])
        tumor_size = float(request.form["tumor_size"])
        treatment_type = int(request.form["treatment_type"])
        diabetes = int(request.form["diabetes"])
        mortality_rate = int(request.form["mortality_rate"])
        insurance_status = int(request.form["insurance_status"])
        cancer_stage = int(request.form["cancer_stage"])
        screening_history = int(request.form["screening_history"])

        input = np.array([[healthcare_cost,tumor_size,treatment_type,diabetes,mortality_rate,insurance_status,cancer_stage,screening_history]])

        scaled_input = scaler.transform(input)

        prediction = model.predict(scaled_input)[0]

        print(prediction)
        return render_template('index.html' , prediction=prediction)
    
    except Exception as e:
        return str(e)
    
if __name__=="__main__":
    app.run(debug=True , host="0.0.0.0" , port=5000)

    
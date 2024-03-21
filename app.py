import pandas as pd
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)
# Đọc dataset từ file không có header
dataset = pd.read_csv('healthcare-dataset-stroke-data.csv', header=None)

# Đặt tên các cột
columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
           'residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']

# Lấy danh sách các giá trị duy nhất ở cột 5, 6 và 9
distinct_values_work_type = dataset[5].unique()
distinct_values_smoking_status = dataset[9].unique()

# Đọc dữ liệu từ label_encoders_dict
label_encoders_dict = joblib.load('label_encoders_dict.pkl')
loaded_model = joblib.load('randomforestmodel.pkl')

# Định tuyến
@app.route('/')
def home():
    return render_template('index.html',
                           work_type_options=distinct_values_work_type,
                           smoking_status_options=distinct_values_smoking_status)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':

        features_encoded = [
            int(request.form['gender']),
            int(request.form['age']),
            int(request.form['hypertension']),
            int(request.form['heart_disease']),
            int(request.form['ever_married']),
            label_encoders_dict['work_type'].transform([request.form['work_type']])[0],
            int(request.form['residence_type']),
            float(request.form['avg_glucose_level']),
            float(request.form['bmi']),
            label_encoders_dict['smoking_status'].transform([request.form['smoking_status']])[0],
        ]
        
        prediction = loaded_model.predict([features_encoded])
        prediction = prediction.item()

        return jsonify({'prediction': "Đột quỵ" if prediction == 1 else "Không đột quỵ"})

if __name__ == '__main__':
    app.run(debug=True)

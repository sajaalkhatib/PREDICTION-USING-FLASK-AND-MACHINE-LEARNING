from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('Project_Model.pkl')
X_columns = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['Get','POST'])
def predict():
    try:
        data = request.form.to_dict()
        print("Received data:", data) 

        features = [data.get(feature, None) for feature in X_columns]
        if None in features:
            missing_features = [X_columns[i] for i in range(len(features)) if features[i] is None]
            return render_template('result.html', error_message=f"Missing features: {', '.join(missing_features)}")

        prediction = model.predict([features])
        result = prediction[0]  
        return render_template('result.html', prediction_text=f'Predicted Class: {result}')
    except Exception as e:
        return render_template('result.html', error_message=str(e))

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('Project_Model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        parents = request.form['parents']
        has_nurs = request.form['has_nurs']
        form = request.form['form']
        children = request.form['children']
        housing = request.form['housing']
        finance = request.form['finance']
        social = request.form['social']
        health = request.form['health']

        
        features = [[parents, has_nurs, form, children, housing, finance, social, health]]
        
        prediction = model.predict(features)[0]

       
        return render_template('result.html', prediction=prediction)
    
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

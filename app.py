from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

app = Flask(__name__)

# Sample data for training the model
data = {
    'CreditScore': [600, 700, 800, 500, 600, 650, 720, 800],
    'Age': [22, 35, 45, 23, 35, 28, 50, 42],
    'Tenure': [1, 2, 3, 1, 2, 1, 4, 5],
    'Balance': [500, 1000, 1500, 0, 500, 100, 2500, 3000],
    'NumOfProducts': [1, 2, 2, 1, 1, 1, 3, 2],
    'HasCrCard': [1, 1, 1, 0, 1, 0, 1, 1],
    'IsActiveMember': [1, 0, 1, 0, 1, 0, 1, 0],
    'EstimatedSalary': [50000, 60000, 70000, 20000, 30000, 40000, 90000, 100000],
    'Churn': [0, 0, 0, 1, 0, 1, 0, 1]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Features and target variable
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train Random Forest model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction_text = ""
    if request.method == 'POST':
        # Get form data
        features = [
            float(request.form['CreditScore']),
            float(request.form['Age']),
            float(request.form['Tenure']),
            float(request.form['Balance']),
            int(request.form['NumOfProducts']),
            int(request.form['HasCrCard']),
            int(request.form['IsActiveMember']),
            float(request.form['EstimatedSalary'])
        ]

        # Load the model
        with open('random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)

        # Make prediction
        prediction = model.predict([features])
        prediction_text = "Churn Probability: {:.2f}".format(model.predict_proba([features])[0][1])
        prediction_text += " | Will Churn: " + ("Yes" if prediction[0] == 1 else "No")

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)

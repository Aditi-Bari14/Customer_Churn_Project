import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your dataset
data = pd.read_csv('customer_data.csv')  # Replace with your dataset path

# Preprocess your dataset
# Assuming the dataset has been preprocessed and 'Churn' is the target column
X = data.drop('Churn', axis=1)  # Features
y = data['Churn']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'random_forest_model.pkl')

# Import necessary libraries
import pandas as pd
import numpy as np
import joblib  # For saving models

# Machine learning libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# For handling class imbalance
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('healthcare-dataset-stroke-data.csv', encoding='utf-8')



# Impute missing values for 'bmi' with median
data['bmi'] = data['bmi'].fillna(data['bmi'].median())

# Drop rows with gender as 'Other' (if any)
data = data[data['gender'] != 'Other']


# Convert categorical text data to lowercase
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

for col in categorical_cols:
    data[col] = data[col].str.lower()

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save the encoder for later use

# Define feature variables and target variable
X = data.drop(['id', 'stroke'], axis=1)
y = data['stroke']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance using SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_scaled, y)

# Split dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, 
                                                    test_size=0.2, 
                                                    random_state=42)

# Initialize Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
classifier.fit(X_train, y_train)

# Predict on test data
y_pred = classifier.predict(X_test)

# Evaluate the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# **Save the trained model, scaler, and label encoders**
joblib.dump(classifier, 'stroke_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
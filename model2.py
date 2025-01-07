import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load data
morning_data = pd.read_csv('morning_session.csv', header=1)  # Training set
afternoon_data = pd.read_csv('afternoon_session.csv', header=1)  # Testing set

# Strip column names to remove any leading/trailing spaces
morning_data.columns = morning_data.columns.str.strip()
afternoon_data.columns = afternoon_data.columns.str.strip()

# Define the activity columns as per the provided list
ACTIVITY_COLUMNS = ['A1', 'Q1', 'A2', 'Q2', 'A3', 'A4', 'Q3', 'Mid', 'AWS Labs', 'Q4', 'A5', 'Q5', 'A6', 'Final']
NUM_ACTIVITIES = len(ACTIVITY_COLUMNS)  # Total activities count
MIN_ACTIVITIES = 5  # Predictions start after the 5th activity

# Convert relevant columns to numeric
def preprocess_data(data):
    # Strip columns to remove any leading/trailing spaces
    data.columns = data.columns.str.strip()

    # Select all the relevant activity columns and final score
    data[ACTIVITY_COLUMNS] = data[ACTIVITY_COLUMNS].apply(pd.to_numeric, errors='coerce')  # Convert to numeric
    data['Final'] = pd.to_numeric(data['Final'], errors='coerce')  # Convert final score to numeric
    data.iloc[:, -1] = pd.to_numeric(data.iloc[:, -1], errors='coerce')  # Convert last column (ultimate final score) to numeric
    return data.fillna(0)  # Replace NaN with 0

morning_data = preprocess_data(morning_data)
afternoon_data = preprocess_data(afternoon_data)

# Function to generate features
def generate_features(data, activity_columns=ACTIVITY_COLUMNS, min_activities=MIN_ACTIVITIES):
    feature_rows = []
    target = []
    
    for _, row in data.iterrows():
        scores = row[activity_columns].values  # Extract activity scores
        ultimate_final_score = row.iloc[-1]  # Access the ultimate final score from the last column
        
        for i in range(min_activities, len(activity_columns) + 1):
            cumulative_score = sum(scores[:i])  # Calculate cumulative score up to i-th activity
            average_score = cumulative_score / i  # Calculate average score up to i-th activity
            feature_rows.append([cumulative_score, average_score, i])
            target.append(ultimate_final_score)
    
    feature_df = pd.DataFrame(
        feature_rows, columns=["cumulative_score", "average_score", "num_activities"]
    )
    return feature_df, np.array(target)

# Generate training and testing features
X_train, y_train = generate_features(morning_data)
X_test, y_test = generate_features(afternoon_data)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'student_score_predictor.pkl')

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Erro:", mae)




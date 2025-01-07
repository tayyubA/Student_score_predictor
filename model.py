import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Step 1: Load the training data (morning session - training set)
train_data_path = "data/course_activity_marks_morning.csv"  # Example path
train_data = pd.read_csv(train_data_path)

# Step 2: Load the test data (afternoon session - test set)
test_data_path = "data/course_activity_marks_afternoon.csv"  # Example path
test_data = pd.read_csv(test_data_path)

# Step 3: Prepare the features (activities) and target (final score)
activity_columns = [f'Activity_{i}' for i in range(1, 15)]  # Activity_1 to Activity_14
X_train = train_data[activity_columns]
y_train = train_data['FinalScore']

# Step 4: Train the model using the morning session data
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Save the trained model to disk
model_filename = "final_score_predictor_model.pkl"
joblib.dump(model, model_filename)
print(f"Model saved as {model_filename}")

# Step 6: Predict final scores for each student in the test set after each activity
def predict_final_scores(model, student_data, activity_columns):
    """
    Predict the final score of a student after each given activity.
    """
    predictions = {}
    for activity_num in range(5, 15):  # Start from 5th activity to 14th
        partial_data = student_data[activity_columns[:activity_num]]  # Select activities up to `activity_num`
        predicted_score = model.predict(partial_data)
        predictions[activity_num] = predicted_score[0]  # Store the predicted score for each activity
    return predictions

# Step 7: Loop over each student in the test set and predict final score after each activity
predictions_per_student = {}

for idx, student in test_data.iterrows():
    student_data = student[activity_columns]
    student_predictions = predict_final_scores(model, student_data, activity_columns)
    predictions_per_student[student['StudentID']] = student_predictions

# Step 8: Output the predictions
for student_id, student_predictions in predictions_per_student.items():
    print(f"Predictions for Student {student_id}:")
    for activity_num, prediction in student_predictions.items():
        print(f"After Activity {activity_num}: Predicted Final Score = {prediction:.2f}")

# Step 9: Optionally, save the predictions to a CSV file
predictions_df = pd.DataFrame.from_dict(predictions_per_student, orient='index')
predictions_df.to_csv("predictions_after_each_activity.csv", index_label="StudentID")
print("Predictions saved to 'predictions_after_each_activity.csv'.")

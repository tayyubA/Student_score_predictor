import gradio as gr
import joblib
import numpy as np

# Load the model
model = joblib.load('student_score_predictor.pkl')

# Define the prediction function
def predict_score(activity_scores):
    # Ensure the input is a list of floats, with at least 5 scores
    activity_scores = np.array(activity_scores, dtype=float)
    if len(activity_scores) < 5:
        return "Please provide at least 5 activities."
    
    cumulative_score = np.sum(activity_scores[:len(activity_scores)])
    average_score = cumulative_score / len(activity_scores)
    features = np.array([[cumulative_score, average_score, len(activity_scores)]])
    
    # Predict final score
    prediction = model.predict(features)
    return prediction[0]

# Create Gradio interface
interface = gr.Interface(
    fn=predict_score,
    inputs=gr.inputs.Dataframe(
        headers=["Activity 1", "Activity 2", "Activity 3", "Activity 4", "Activity 5", "Activity 6", "Activity 7", "Activity 8", "Activity 9", "Activity 10", "Activity 11", "Activity 12", "Activity 13", "Activity 14"],
        default=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        type="numpy"
    ),
    outputs="text"
)

# Launch the interface
interface.launch()

import joblib
import pandas as pd

# Load the trained model
model_path = "C:/Users/savan/Desktop/hackaton/results/decision_tree_tuned_model.joblib"  # Replace with actual path to your model
logistic_regression_model = joblib.load(model_path)

# Load the label encoder to interpret the model's output
label_encoder_path = "C:/Users/savan/Desktop/hackaton/label_encoder.joblib"  # Replace with actual path to your label encoder
label_encoder = joblib.load(label_encoder_path)

# Get the list of symptoms (feature names) from the model
try:
    symptom_names = logistic_regression_model.feature_names_in_
except AttributeError:
    raise ValueError("Model does not contain feature names. Ensure it was trained on a DataFrame with named columns.")

# Collect symptom data from the user
print("Please enter the following symptoms. Use 1 for 'Yes' and 0 for 'No'.")
user_input = {}
for symptom in symptom_names:
    while True:
        try:
            value = int(input(f"{symptom}: "))
            if value in [0, 1]:
                user_input[symptom] = value
                break
            else:
                print("Please enter 1 for 'Yes' or 0 for 'No'.")
        except ValueError:
            print("Invalid input. Please enter a numeric value: 1 or 0.")

# Convert the user input into a DataFrame
input_data = pd.DataFrame([user_input])

# Align input_data with the model's expected feature names
input_data_aligned = input_data.reindex(columns=symptom_names, fill_value=0)

# Make a prediction with confidence
logistic_prediction = logistic_regression_model.predict(input_data_aligned)
prediction_proba = logistic_regression_model.predict_proba(input_data_aligned)

# Get the index of the predicted class and the confidence
predicted_index = logistic_prediction[0]
confidence_percentage = prediction_proba[0][predicted_index] * 100

# Decode the predicted result back to its original label
predicted_prognosis = label_encoder.inverse_transform(logistic_prediction)

# Output the prediction and confidence
print("\nPredicted prognosis:", predicted_prognosis[0])
print(f"Confidence of this prediction: {confidence_percentage:.2f}%")
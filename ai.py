import pandas as pd
import joblib

# Load the new dataset (replace with the correct file path)
rndforest_data = pd.read_csv('C:\\Users\\savan\\Desktop\\hackaton\\data_0.csv')
dcstree_data = pd.read_csv('C:\\Users\\savan\\Desktop\\hackaton\\data_1.csv')
mlp_data = pd.read_csv('C:\\Users\\savan\\Desktop\\hackaton\\data_2.csv')

# Load the trained models
random_forest_model = joblib.load('C:/Users/savan/Desktop/hackaton/results/random_forest_model.joblib')
decision_tree_model = joblib.load('C:/Users/savan/Desktop/hackaton/results/decision_tree_model.joblib')
mlp_model = joblib.load('C:/Users/savan/Desktop/hackaton/results/multi-layer_perceptron_model.joblib')

# Load the label encoder if needed to decode predictions
label_encoder = joblib.load('C:/Users/savan/Desktop/hackaton/label_encoder.joblib')

# Make predictions with each model on the new dataset and get probabilities
random_forest_probs = random_forest_model.predict_proba(rndforest_data)
decision_tree_probs = decision_tree_model.predict_proba(dcstree_data)
mlp_probs = mlp_model.predict_proba(mlp_data)

# Get the class with the highest probability and its confidence percentage
random_forest_predictions = label_encoder.inverse_transform(random_forest_probs.argmax(axis=1))
random_forest_confidence = random_forest_probs.max(axis=1) * 100  # confidence in percentage

decision_tree_predictions = label_encoder.inverse_transform(decision_tree_probs.argmax(axis=1))
decision_tree_confidence = decision_tree_probs.max(axis=1) * 100  # confidence in percentage

mlp_predictions = label_encoder.inverse_transform(mlp_probs.argmax(axis=1))
mlp_confidence = mlp_probs.max(axis=1) * 100  # confidence in percentage

# Store predictions and confidences in a DataFrame to compare results
predictions_df = pd.DataFrame({
    'Random Forest Prediction': random_forest_predictions,
    'Random Forest Confidence (%)': random_forest_confidence,
    'Decision Tree Prediction': decision_tree_predictions,
    'Decision Tree Confidence (%)': decision_tree_confidence,
    'Multi-layer Perceptron Prediction': mlp_predictions,
    'MLP Confidence (%)': mlp_confidence
})

# Save predictions with confidence to a new CSV file
predictions_df.to_csv('C:/Users/savan/Desktop/hackaton/predictions_with_confidence.csv', index=False)

print("Predictions with confidence saved to predictions_with_confidence.csv")

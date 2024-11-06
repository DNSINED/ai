import pandas as pd
import joblib

# Load the new dataset (replace with the correct file path)
rndforest_data = pd.read_csv('C:\\Users\\savan\\Desktop\\hackaton\\data_0.csv')
dcstree_data = pd.read_csv('C:\\Users\\savan\\Desktop\\hackaton\\data_1.csv')
mlp_data = pd.read_csv('C:\\Users\\savan\\Desktop\\hackaton\\data_2.csv')

# Preprocess the new data to match the training data
# Encode categorical variables or apply any scaling if needed
# Example: If you used LabelEncoder or StandardScaler on your training data, apply the same here.

# Load the trained models
random_forest_model = joblib.load('C:/Users/savan/Desktop/hackaton/results/random_forest_model.joblib')
decision_tree_model = joblib.load('C:/Users/savan/Desktop/hackaton/results/decision_tree_model.joblib')
mlp_model = joblib.load('C:/Users/savan/Desktop/hackaton/results/multi-layer_perceptron_model.joblib')

# Load the label encoder if needed to decode predictions
label_encoder = joblib.load('C:/Users/savan/Desktop/hackaton/label_encoder.joblib')

# Make predictions with each model on the new dataset
random_forest_predictions = random_forest_model.predict(rndforest_data)
decision_tree_predictions = decision_tree_model.predict(dcstree_data)
mlp_predictions = mlp_model.predict(mlp_data)

# Convert numerical predictions back to disease labels (if encoded)
random_forest_predictions = label_encoder.inverse_transform(random_forest_predictions)
decision_tree_predictions = label_encoder.inverse_transform(decision_tree_predictions)
mlp_predictions = label_encoder.inverse_transform(mlp_predictions)

# Store predictions in a DataFrame to compare results
predictions_df = pd.DataFrame({
    'Random Forest': random_forest_predictions,
    'Decision Tree': decision_tree_predictions,
    'Multi-layer Perceptron': mlp_predictions
})

# Save predictions to a new CSV file
predictions_df.to_csv('C:/Users/savan/Desktop/hackaton/predictions_on_new_data.csv', index=False)

print("Predictions saved to predictions_on_new_data.csv")

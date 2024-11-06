import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import joblib

# Load the SymbiPredict dataset
data = pd.read_csv("C:\\Users\\savan\\Desktop\\hackaton\\gaussianNB_symbipredict_2022.csv")

# Encode the target label 'prognosis'
label_encoder = LabelEncoder()
data['prognosis'] = label_encoder.fit_transform(data['prognosis'])  # Encode target labels

# Separate features and target variable
X = data.drop(columns=['prognosis'])
y = data['prognosis']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifiers
classifiers = {
    # "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    # "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    # "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Gaussian Naive Bayes": GaussianNB(),
    # "Multi-layer Perceptron": MLPClassifier(max_iter=2000, random_state=42)
}

# Dictionary to store the results for each classifier
results = {}

# Train and evaluate each classifier
for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_train, y_train)
    
    # Save each model
    model_path = f"C:\\Users\\savan\\Desktop\\hackaton\\results\\{name.replace(' ', '_').lower()}_model.joblib"
    joblib.dump(clf, model_path)
    
    # Make predictions and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Store results in dictionary
    results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    
    # Print metrics
    print(f"\n{name} Accuracy: {accuracy * 100:.2f}%")
    print(f"{name} Precision: {precision * 100:.2f}%")
    print(f"{name} Recall: {recall * 100:.2f}%")
    print(f"{name} F1 Score: {f1 * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print("\n" + "="*50 + "\n")

# Display a summary of results for all models
print("Comparison of Model Metrics:")
for model, metrics in results.items():
    print(f"{model} -> Accuracy: {metrics['Accuracy'] * 100:.2f}%, Precision: {metrics['Precision'] * 100:.2f}%, "
          f"Recall: {metrics['Recall'] * 100:.2f}%, F1 Score: {metrics['F1 Score'] * 100:.2f}%")

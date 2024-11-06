#all above 0.01
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Load the SymbiPredict dataset
data = pd.read_csv('C:/Users/savan/Desktop/hackaton/symbipredict_2022.csv')

# Encode the target label 'prognosis'
label_encoder = LabelEncoder()
data['prognosis'] = label_encoder.fit_transform(data['prognosis'])

# Separate features and target variable
X = data.drop(columns=['prognosis'])
y = data['prognosis']
model = DecisionTreeClassifier()
# fit the model
model.fit(X, y)
# get importance
importance = model.feature_importances_
# summarize feature importance

feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': importance})
# Sort features by absolute coefficient value in descending order and select top 10
top_10_features = feature_importance.reindex(feature_importance.Coefficient.abs().sort_values(ascending=False).index)[:38]
# Print the top 10 features with their coefficients
print("Top 10 most important features and their coefficients:")
print(top_10_features)

# Plot feature importance for top 10 features
plt.figure(figsize=(10, 6))
plt.barh(top_10_features['Feature'], top_10_features['Coefficient'], color='skyblue')
plt.xlabel('Coefficient')
plt.title('Top 10 Most Important Features by Coefficient')
plt.gca().invert_yaxis()  # Invert y-axis to show highest importance at the top
plt.show()

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder

# Load your dataset
data = pd.read_csv('C:/Users/savan/Desktop/hackaton/symbipredict_2022.csv')

# Convert target variable 'prognosis' from string to numeric
label_encoder = LabelEncoder()
data['prognosis'] = label_encoder.fit_transform(data['prognosis'])

# Separate features and target variable
X = data.drop('prognosis', axis=1)
y = data['prognosis']

# Ensure all features are numeric; if not, apply encoding or transformation
X = X.apply(pd.to_numeric, errors='coerce')

# Define and compile the Keras MLP model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(len(np.unique(y)), activation='softmax')  # Assuming y is categorical
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(X, y, epochs=10, batch_size=32, verbose=1)

# Weight-based importance
weights = model.layers[0].get_weights()[0]  # Get weights from the first layer
importance_weights = weights.mean(axis=1)

# Gradient-based importance using TensorFlow
input_tensor = tf.convert_to_tensor(X.values, dtype=tf.float32)

with tf.GradientTape() as tape:
    tape.watch(input_tensor)
    output = model(input_tensor)  # Get model output (logits)

# Compute gradients with respect to the input features
gradients = tape.gradient(output, input_tensor)  # Get gradients for logits
importance_gradients = np.mean(np.abs(gradients.numpy()), axis=0)

# Combine and display results
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Weight Importance': importance_weights,
    'Gradient Importance': importance_gradients
})

# Set pandas options to display all rows and columns
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

# Print the entire DataFrame
im = importance_df[importance_df['Gradient Importance'] > 4e-08]
print(im.sort_values(by='Gradient Importance', ascending=False))

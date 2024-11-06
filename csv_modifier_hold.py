import pandas as pd

# Load the original data
file_path = "C:\\Users\\savan\\Desktop\\hackaton\\symbipredict_2022.csv"  # Replace this with the correct path if necessary
data = pd.read_csv(file_path)

# List of all symptoms (excluding the target column 'prognosis')
symptoms = [col for col in data.columns if col != 'prognosis']

# Display symptoms to the user
print("Here is the list of symptoms:\n")
for i, symptom in enumerate(symptoms, start=1):
    print(f"{i}. {symptom}")

# Ask user to select columns to keep
print("\nEnter the numbers of the symptoms you want to keep, separated by commas (e.g., 1, 3, 5):")
keep_indices = input()
keep_indices = [int(i.strip()) - 1 for i in keep_indices.split(",")]

# Get the selected columns to keep based on indices
columns_to_keep = [symptoms[i] for i in keep_indices]
columns_to_keep.append('prognosis')  # Always keep the target column

# Keep only the selected columns
modified_data = data[columns_to_keep]

# Save the modified dataset
output_path = "C:\\Users\\savan\\Desktop\\hackaton\\data_test_symbipredict_2022.csv"
modified_data.to_csv(output_path, index=False)
print(f"Modified data saved to {output_path} with the following columns kept: {columns_to_keep}")

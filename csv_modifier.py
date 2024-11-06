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

# Ask user to select columns to remove
print("\nEnter the numbers of the symptoms you want to remove, separated by commas (e.g., 1, 3, 5):")
remove_indices = input()
remove_indices = [int(i.strip()) - 1 for i in remove_indices.split(",")]

# Get the selected columns to remove based on indices
columns_to_remove = [symptoms[i] for i in remove_indices]

# Remove the selected columns
modified_data = data.drop(columns=columns_to_remove)

# Save the modified dataset
output_path = "C:\\Users\\savan\\Desktop\\hackaton\\new_symbipredict_2022.csv"
modified_data.to_csv(output_path, index=False)
print(f"Modified data saved to {output_path} with the following columns removed: {columns_to_remove}")

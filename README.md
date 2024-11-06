MedEcho

# AI Model Prediction Script

This project provides a script to load and use pre-trained machine learning models to make predictions on new datasets. The models and datasets need to be correctly formatted and saved in specified locations.

## Table of Contents

1. [Project Description](#project-description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [File Descriptions](#file-descriptions)
5. [Dataset](#dataset)
6. [Configuration](#configuration)
7. [Examples](#examples)
8. [Contributing](#contributing)
9. [License](#license)

---

## Project Description

This repository contains scripts to load, preprocess, and generate predictions using three pre-trained models:
- **Random Forest**
- **Decision Tree**
- **Multi-Layer Perceptron (MLP)**

Each model predicts based on different datasets provided. The predictions are intended for tasks requiring classification or regression analysis, depending on the dataset and model used.

## Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/DNSINED/MedEcho.git
   ```
2. **Install dependencies**:
   Make sure you have Python installed, then install necessary libraries:
   ```bash
   pip install pandas joblib
   ```

3. **Organize Datasets**:
   Place your datasets in the designated paths as specified in the script, or update the script with the correct paths.

## Usage

To run the model and generate predictions:
1. Ensure your datasets and model files are placed correctly.
2. Run the script:
   ```bash
   python ai.py
   ```

## File Descriptions

- **ai.py**: This script loads the datasets and pre-trained models, preprocesses data if needed, and performs predictions.
- **model_trainer.py**: Use this script to train models if not already trained. Make sure to configure the training parameters as required (details below).

## Dataset

The script uses three datasets:
1. `data_0.csv`: For the Random Forest model.
2. `data_1.csv`: For the Decision Tree model.
3. `data_2.csv`: For the MLP model.

These datasets should have the same features used in the training data to ensure compatibility.

## Configuration

- **Model Paths**: Update the paths to the model files in `ai.py` if they are stored in different directories.
- **Dataset Paths**: Update the paths for the CSV files in `ai.py` to match the locations of your data files.

## Examples

To run a sample prediction:
```bash
python ai.py
```

The script will output predictions based on the input data.

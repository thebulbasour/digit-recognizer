
# Digit Recognizer

A handwritten digit classification project using Machine Learning and Deep Learning. The goal is to recognize digits from 0 to 9 using the Kaggle Digit Recognizer dataset.

## Project Overview

This project explores handwritten digit recognition using two models:

- Logistic Regression — a Machine Learning baseline.
- Convolutional Neural Network (CNN) — a Deep Learning model designed for image recognition.

The models are trained and evaluated using Python, Scikit-learn, and TensorFlow.

## Dataset

The dataset is from the [Kaggle Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer).

- Training data: 42,000 labeled images
- Test data: 28,000 unlabeled images
- Image dimensions: 28 × 28 pixels
- Pixel values: 0–255

Each image is represented by 784 pixel values.

## Project Workflow

1. Load and explore the dataset.
2. Separate image features and digit labels.
3. Normalize pixel values to the range 0–1.
4. Split the training data into training and validation sets.
5. Train a Logistic Regression baseline.
6. Build and train a CNN.
7. Compare model performance.
8. Generate predictions for the Kaggle test dataset.
9. Create and submit `submission.csv`.

## Models and Results

| Model | Validation Accuracy |
|---|---:|
| Logistic Regression | 91.31% |
| CNN | 98.88% |

The CNN achieved a Kaggle submission score of **0.98810**.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook
- Git and GitHub

## Project Structure

```text
digit-recognizer/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── digit_recognizer.ipynb
├── digit_recognizer_cnn.keras
├── model_results.csv
├── submission.csv
├── .gitignore
└── README.md
```

The Kaggle dataset files are stored locally and excluded from Git tracking.

## How to Run

1. Clone this repository.
2. Install the required Python packages.
3. Download the dataset from Kaggle.
4. Place the CSV files inside the `data/` folder.
5. Open `digit_recognizer.ipynb` in Jupyter Notebook or VS Code.
6. Run the notebook cells in order.

Install the required packages using:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow jupyter
```

## What I Learned

- Data exploration and preprocessing
- Image representation using pixel values
- Training and evaluating Machine Learning models
- Building a CNN using TensorFlow and Keras
- Generating Kaggle competition submissions
- Using Git and GitHub for version control

## Kaggle Competition

[Digit Recognizer — Kaggle](https://www.kaggle.com/competitions/digit-recognizer)
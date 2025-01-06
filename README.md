# neural-network-challenge-1
Module 18 Neural Network Challenge

# Student Loan Risk Prediction with Deep Learning

## Description
This project predicts the likelihood of student loan repayment using a neural network model. By analyzing historical data, the model assists lenders in offering personalized interest rates based on borrowers' profiles. The project explores deep learning and Random Forest models for predictive analytics.

Key objectives include:
1. Predicting student loan repayment likelihood based on borrower data.
2. Comparing the performance of neural networks and Random Forest models.
3. Discussing potential enhancements, including a recommendation system for student loans.

---

## Table of Contents
- [Description](#description)
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## Dataset Description
The dataset used for this project includes information about past student loan recipients. The following features are included:
- **Demographics**: Age, location, and school information.
- **Academic Data**: GPA, degree type (e.g., STEM vs. non-STEM), and test scores.
- **Financial Information**: Household income, existing debt, and financial aid eligibility.
- **Loan Details**: Loan history, requested amounts, and repayment status.
- **Target Variable**: `credit_ranking`, indicating the borrower's likelihood of repayment.

The dataset was preprocessed to handle missing values, scale features, and split into training/testing sets.

---

## Installation
1. Clone the repository to your local environment:

`git clone https://github.com/Pete1001/neural-network-challenge-1.git`

2. Install required Python libraries:

`pip install -r requirements.txt`

3. Open the Jupyter Notebook in Google Colab or a local environment.

4. Launch the Jupyter Notebook in Google Colab or a local environment:
- Upload the `student_loans_with_deep_learning.ipynb` file to Google Colab if needed.

---

4. Launch the Jupyter Notebook in Google Colab or a local environment:
- Upload the `student_loans_with_deep_learning.ipynb` file to Google Colab if needed.

---

## Usage
1. **Data Preparation**: Follow the notebook instructions to preprocess the data, including handling missing values, feature scaling, and splitting the data into training and testing sets.
2. **Model Training**: Use TensorFlow to create, compile, and train a neural network model. The model predicts the probability of successful loan repayment based on borrower features.
3. **Model Evaluation**: Compare the neural network's performance against a Random Forest model for insights into accuracy and prediction reliability.
4. **Making Predictions**: Utilize the saved `.keras` model to make predictions on new datasets.
5. **Analyze Results**: Examine classification reports, confusion matrices, and model performance metrics.

---

## Files
- `student_loans_with_deep_learning.ipynb`: The main Jupyter Notebook file that contains all steps for data preprocessing, model training, evaluation, and testing.
- `student_loans.keras`: The trained neural network model saved for deployment and reuse.

---

## Model Architecture
The deep neural network model was structured as follows:
- **Input Layer**: Accepts 10 input features, representing borrower details and financial data.
- **Hidden Layers**:
- Layer 1: 6 neurons, ReLU activation function, processes initial patterns.
- Layer 2: 3 neurons, ReLU activation function, extracts deeper insights.
- **Output Layer**: 1 neuron with a Sigmoid activation function, predicts the probability of loan repayment as a binary outcome (0 or 1).

The model was trained for 100 epochs using the binary crossentropy loss function, Adam optimizer, and accuracy as the evaluation metric.
From the attached graphs, we saw that accuracy was still increasing after 50 epochs; thus, we increased the number of epochs to 100.

---

## Results
### Neural Network Model:
- **Accuracy**: 85%
- **Loss**: Low, indicating good model convergence.
- **Confusion Matrix**: Visualized in the notebook for further analysis.

### Random Forest Model:
- **Accuracy**: 90% (outperformed the neural network on this dataset).
- **Precision/Recall/F1-Score**: Higher across all metrics compared to the neural network.

The Random Forest model proved to be a simpler yet more effective approach for this dataset, but the neural network remains a viable solution for future iterations.

---

## Future Enhancements
1. **Feature Expansion**: Include additional borrower features such as loan amount, interest rate, and repayment duration to improve prediction accuracy.
2. **Hyperparameter Tuning**: Experiment with different neural network architectures, learning rates, and batch sizes for optimization such as the Keras Tuner.
3. **Recommendation System**: Build a recommendation engine for student loans, leveraging the trained model to suggest optimal loan products tailored to individual borrower profiles.
4. **Deployment**: Develop an API or web interface to allow lenders to utilize the model for real-time loan assessment.

---

## License
This project is licensed under the MIT License. by Pete Link. You are free to use, modify, and distribute this project with proper attribution.
https://www.linkedin.com/in/petelink/
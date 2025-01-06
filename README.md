# neural-network-challenge-1
Module 18 Neural Network Challenge

# Student Loan Risk Prediction with Deep Learning

## Description
This project predicts the likelihood of student loan repayment using a neural network model. By analyzing historical data, the model assists lenders in offering personalized interest rates based on borrowers' profiles. The project explores deep learning and random forest models for predictive analytics.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [License](#license)

## Installation
1. Clone the repository:

`git clone https://github.com/username/neural-network-challenge-1.git`

2. Install required Python libraries:

`pip install -r requirements.txt`

3. Open the Jupyter Notebook in Google Colab or a local environment.


4. Launch the Jupyter Notebook in Google Colab or a local environment:
- Upload the `student_loans_with_deep_learning.ipynb` file to Google Colab if needed.

---

## Usage
1. **Data Preparation**: Follow the notebook instructions to preprocess the data, including handling missing values, feature scaling, and splitting the data into training and testing sets.
2. **Model Training**: Use TensorFlow to create, compile, and train a neural network model. The model predicts the probability of successful loan repayment based on borrower features.
3. **Model Evaluation**: Compare the neural network's performance against a Random Forest model for insights into accuracy and prediction reliability.
4. **Making Predictions**: Utilize the saved `.keras` model to make predictions on new datasets.
5. **Analyze Results**: Examine classification reports and visualization of model performance metrics.

---

## Files
- `student_loans_with_deep_learning.ipynb`: The main Jupyter Notebook file that contains all steps for data preprocessing, model training, evaluation, and testing.
- `student-loans.csv`: The dataset containing historical loan recipient data, including features such as academic performance, loan details, and repayment history.
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

---

## Results
The project achieved the following results:
- **Neural Network Model**:
- Accuracy: 85%
- Loss: Low, indicating good model convergence.
- **Random Forest Model**:
- Accuracy: 90%, outperforming the neural network in this dataset.

The Random Forest model proved to be a simpler yet more effective approach for this dataset, but the neural network remains a viable solution for future iterations.

---

## Future Enhancements
- **Feature Expansion**: Include additional borrower features such as loan amount, interest rate, and repayment duration to improve prediction accuracy.
- **Hyperparameter Tuning**: Experiment with different neural network architectures, learning rates, and batch sizes for optimization.
- **Recommendation System**: Build a recommendation engine for student loans, leveraging the trained model to suggest optimal loan products tailored to individual borrower profiles.
- **Deployment**: Develop an API or web interface to allow lenders to utilize the model for real-time loan assessment.

---

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute this project with proper attribution.

## License
This project is licensed under the MIT License by Pete Link
https://www.linkedin.com/in/petelink/
# 📊 Customer Churn Prediction Using Machine Learning

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to leave a company. Built using Python, Scikit-learn, and Streamlit, this project combines a trained Gradient Boosting model with an interactive web application to estimate customer churn probability.

<p align="center">
  <a href="https://customer-churn-prediction-ml-aquasa-aziz.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Streamlit-red?style=for-the-badge&logo=streamlit" alt="Live Demo">
  </a>
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange?style=for-the-badge" alt="Machine Learning">
</p>

## 🚀 Live Application

**Try the deployed application here:**

🔗 [Customer Churn Prediction App](https://customer-churn-prediction-ml-aquasa-aziz.streamlit.app/)

---

## 📌 Project Overview

Customer churn is a major challenge for telecommunication companies. When customers discontinue their services, businesses lose recurring revenue and may face additional customer acquisition costs.

This project uses historical telecom customer data to build a machine learning model that identifies customers who are likely to churn.

Users can enter customer details through an interactive Streamlit application and receive a prediction along with the estimated churn probability.

## 🎯 Project Objectives

- Understand and explore telecom customer churn data.
- Perform data cleaning and preprocessing.
- Apply feature engineering to create meaningful features.
- Train and evaluate machine learning models.
- Select a suitable model for customer churn prediction.
- Build an interactive web application using Streamlit.
- Deploy the trained model for real-world accessibility.

## ✨ Key Features

- **Interactive User Interface:** Enter customer information through a Streamlit web application.
- **Churn Prediction:** Classifies customers as likely to churn or likely to stay.
- **Probability Estimation:** Displays the estimated churn probability.
- **Customer Risk Profiles:** Includes sample high-risk and low-risk customer profiles.
- **Data Preprocessing:** Applies consistent encoding, feature engineering, and scaling.
- **Visual Analytics:** Displays prediction probabilities and relevant risk information.
- **Saved Model Artifacts:** Loads the trained model and fitted preprocessor without retraining.
- **Cloud Deployment:** Accessible online through Streamlit Community Cloud.

## 🧠 Machine Learning Workflow

The project follows an end-to-end machine learning pipeline:

1. **Data Collection:** Load the telecom customer churn dataset.
2. **Exploratory Data Analysis (EDA):** Understand customer characteristics, feature distributions, and churn patterns.
3. **Data Cleaning:** Handle missing values and prepare the dataset.
4. **Data Preprocessing:**
   - Encode categorical variables.
   - Convert binary features into numerical values.
   - Scale numerical features.
5. **Feature Engineering:**
   - Create customer tenure groups.
   - Calculate average monthly charges based on tenure.
   - Calculate the number of services used by each customer.
6. **Model Training:** Train and compare machine learning algorithms.
7. **Model Evaluation:** Evaluate models using classification performance metrics.
8. **Model Persistence:** Save the selected model and fitted preprocessing object using Joblib.
9. **Application Development:** Integrate the trained model into Streamlit.
10. **Deployment:** Deploy the application using Streamlit Community Cloud.

## 🤖 Selected Machine Learning Model

### Gradient Boosting Classifier

The selected model for this application is the **Gradient Boosting Classifier** from Scikit-learn.

Gradient Boosting is an ensemble learning technique that combines multiple weak learners, typically decision trees, to build a stronger predictive model.

The model performs binary classification:

| Class | Prediction               |
| ----- | ------------------------ |
| 0     | Customer Likely to Stay  |
| 1     | Customer Likely to Churn |

The application also displays prediction probabilities to help users understand the model's confidence in its prediction.

## 🛠️ Technologies Used

| Category                | Technologies              |
| ----------------------- | ------------------------- |
| Programming Language    | Python                    |
| Data Manipulation       | Pandas, NumPy             |
| Machine Learning        | Scikit-learn              |
| Data Visualization      | Plotly                    |
| Web Application         | Streamlit                 |
| Model Persistence       | Joblib                    |
| Development Environment | Jupyter Notebook, VS Code |
| Version Control         | Git, GitHub               |
| Deployment              | Streamlit Community Cloud |

## 📂 Project Structure

```text
Customer-Churn-Prediction-ML/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── telco_comm_churn.csv
│
├── models/
│   ├── best_model_gradient_boosting.pkl
│   └── preprocessor.pkl
│
├── src/
│   └── data_preprocessing.py
│
└── notebooks/
    └── Jupyter Notebooks for EDA,
        preprocessing and model training
```

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the Project Directory

```bash
cd Customer-Churn-Prediction-ML
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your default browser.

Local application URL:

```text
http://localhost:8501
```

**Note:** Use a compatible Python version and the dependencies specified in `requirements.txt` to avoid model compatibility issues.

## 📈 How to Use the Application

1. Open the live application.
2. Enter the customer's demographic information.
3. Provide account and service-related details.
4. Select the customer's contract and payment information.
5. Submit the customer details for prediction.
6. View the predicted churn category and probability.
7. Explore the visual results and customer risk information.

## 💾 Model Artifacts

The project contains two important saved model files:

| File                               | Description                                                            |
| ---------------------------------- | ---------------------------------------------------------------------- |
| `best_model_gradient_boosting.pkl` | Stores the trained Gradient Boosting Classifier.                       |
| `preprocessor.pkl`                 | Stores the fitted preprocessing object, including encoders and scaler. |

These artifacts ensure that new customer data is transformed consistently with the data used during model training.

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

Deployment workflow:

1. Push the project files to GitHub.
2. Connect the GitHub repository to Streamlit Community Cloud.
3. Configure the application entry point as `app.py`.
4. Install the dependencies from `requirements.txt`.
5. Deploy the application.

🔗 **Live Application:** [Open Customer Churn Predictor](https://customer-churn-prediction-ml-aquasa-aziz.streamlit.app/)

## 🔮 Future Improvements

- Compare additional machine learning algorithms.
- Perform advanced hyperparameter tuning.
- Improve model performance through feature selection.
- Integrate SHAP for model explainability.
- Build an advanced customer churn analytics dashboard.
- Add customer segmentation.
- Generate personalized customer retention recommendations.

## 👨‍💻 Author

**Aquasa Aziz**

Aspiring Machine Learning and Data Science Professional.

Currently developing practical skills in Python, Data Analysis, and Machine Learning through hands-on projects.

- GitHub: [https://github.com/AquasaAziz247](https://github.com/AquasaAziz247)
- LinkedIn: [https://www.linkedin.com/in/aquasa-aziz/](https://www.linkedin.com/in/aquasa-aziz/)

---

## ⭐ Support

If you find this project useful or interesting, consider giving the repository a ⭐ on GitHub!

**Built with Python, Machine Learning, and Streamlit.**

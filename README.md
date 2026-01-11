# 🩺 Diabetes Disease Prediction Using AutoML

This project builds an end-to-end machine learning system for predicting whether a patient is diabetic using medical attributes. It uses **FLAML (Fast Lightweight AutoML)** to automatically select the best-performing classification model, and provides an interactive **Gradio web interface** for making predictions.

---

## 📌 Project Highlights

- Automated model selection and tuning using FLAML  
- Data cleaning and preprocessing pipeline  
- Feature scaling using `StandardScaler`  
- Model evaluation and persistence  
- Interactive Gradio-based web application  
- Ready for Jupyter, Colab, and local execution  

---

## 📂 Project Structure
├── diabetes.csv
├── Diabetis disease prediction using auto ml.ipynb
├── diabetes_automl_model.pkl
├── diabetes_scaler.pkl
├── bg.jpeg
├── README.md
---

## 📊 Dataset

- File: `diabetes.csv`  
- Target column: `Outcome` (0 = Non-Diabetic, 1 = Diabetic)

### Input Features

| Feature | Description |
|--------|------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Triceps skin fold thickness |
| Insulin | 2-hour serum insulin |
| BMI | Body Mass Index |
| DiabetesPedigreeFunction | Genetic risk |
| Age | Age of the patient |

---

## ⚙️ Technologies Used

- Python 3.10+  
- Pandas, NumPy  
- Scikit-learn  
- FLAML (AutoML)  
- Joblib  
- Gradio  
- Jupyter Notebook  

---

## 🧪 Data Preprocessing

- Replaced invalid zero values in medical features with NaN  
- Filled missing values using median imputation  
- Standardized features using `StandardScaler`  
- Stratified train-test split (80/20)  

---

## 🤖 Model Training

FLAML automatically searches for the best model using:

```python
automl_settings = {
    "time_budget": 60,
    "metric": "accuracy",
    "task": "classification",
    "log_file_name": "flaml_diabetes.log",
    "n_jobs": -1,
}

📈 Evaluation

The model is evaluated using:

Accuracy Score

Classification Report

Confusion Matrix

💾 Saved Artifacts

diabetes_automl_model.pkl — trained model

diabetes_scaler.pkl — fitted scaler

These are loaded directly by the Gradio app for inference.

🌐 Gradio Web Interface

The project includes a Gradio UI where users can input patient details and receive a diabetes prediction with probability.

Run the interface and open it in your browser to interact with the model.

🚀 How to Run
Option 1: Local Machine
git clone https://github.com/pratibha-singh13/Diabetes-Prediction-using-Auto-ML.git
cd Diabetes-Prediction-using-Auto-ML
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install flaml pandas scikit-learn joblib gradio numpy==1.26.4
jupyter notebook


Open Diabetis disease prediction using auto ml.ipynb and run all cells.

# 🩺 Heart Disease Risk Prediction System

An interactive web application built with **Streamlit** and **SHAP** for clinical heart disease risk prediction. This project was developed as the final assignment for the **Categorical Data Analysis** course.

The system processes dynamic patient clinical inputs, applies real-time preprocessing, and outputs a calibrated risk probability alongside local model interpretability.

## 🚀 Features
- **Real-Time Feature Pipeline**: Automatically syncs web inputs with the training data preparation.
- **Explainable AI (XAI)**: Integrates `shap.TreeExplainer` to provide patient-specific feature contribution visualizations via interactive Force Plots and sorted Bar Plots.
- **Clinical Decision Support**: Generates targeted risk stratification and medical recommendations based on predicted probabilities and key clinical indicators.

## 📂 Repository Structure
- `app.py`: The core Streamlit web application script.
- `heart_model3.pkl`: The trained machine learning classification model.
- `scaler.pkl`: The saved `StandardScaler` from the training notebook to normalize numeric inputs.
- `background_data.csv`: Reference dataset utilized by the SHAP explainer for baseline calculations.
- `requirements.txt`: List of dependencies required for cloud deployment.
- `code_flow.ipynb`: The whole code flow for this project (only for local).

## 🛠️ Local Installation & Running
To build the streamlit locally, clone the repository, install the dependencies, and start the Streamlit server:

```bash
pip install -r requirements.txt
streamlit run app.py

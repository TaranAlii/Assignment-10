# ❤️ AI-ML Assignment 10: Heart Disease Risk Assessment API & Dashboard

This repository contains the implementation of **AI-ML Assignment – 10**, which focuses on developing and deploying a **Machine Learning-based Heart Disease Risk Assessment System**. The project combines a **Random Forest Classifier**, a **Flask REST API**, and a **responsive web dashboard** to predict the likelihood of heart disease using patient clinical information.

The trained model has been deployed on **Render**, allowing users to access predictions through both a user-friendly web interface and a REST API.

---

# 🌐 Live Demo

**Render Deployment:**

https://assignment-10-5bzy.onrender.com/

---

# 👤 Student Information

- **Name:** Taran Ali Ahmed
- **Registration Number:** 23BCE10952
- **Application Number:** IN26009846
- **Batch:** 2(B)

---

# 📌 Objective

The objective of this assignment is to design, develop, and deploy a complete Machine Learning application capable of predicting heart disease risk from patient clinical data.

The project covers the complete ML deployment lifecycle, including:

- Data preprocessing and exploratory analysis
- Machine learning model development
- Model evaluation
- Model serialization
- REST API development using Flask
- Responsive web dashboard
- Cloud deployment using Render

The final application enables healthcare professionals and users to receive real-time predictions through a web browser or API requests.

---

# 📊 Dataset

The project uses the **Heart Disease Dataset**.

### Dataset Source

**Kaggle Dataset**

https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

---

## Dataset Summary

| Item | Value |
|------|-------:|
| Total Records | 1,025 |
| Features | 13 |
| Target Variable | Heart Disease (0 / 1) |
| Missing Values | 0 |

---

## Features Used

| Feature | Description |
|---------|-------------|
| age | Age of the patient |
| sex | Gender (1 = Male, 0 = Female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting ECG result |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | Slope of ST segment |
| ca | Number of major vessels |
| thal | Thalassemia |

### Target

- **1 → Heart Disease Detected**
- **0 → No Heart Disease**

---

# 🛠️ Technologies & Libraries

The project was developed using:

- Python 3.11
- Flask
- Scikit-Learn
- Pandas
- NumPy
- Joblib
- Gunicorn
- HTML5
- CSS3
- Bootstrap

---

# ⚙️ Methodology

## 1. Data Understanding

The dataset was first explored to understand its structure.

The following tasks were performed:

- Loaded dataset
- Checked data types
- Displayed summary statistics
- Verified missing values
- Identified input and target variables

Dataset statistics:

- **1025 patient records**
- **13 clinical features**
- **No missing values**

---

## 2. Data Preprocessing

The following preprocessing steps were applied:

- Removed unnecessary inconsistencies
- Verified data quality
- Selected input features
- Split dataset into

| Dataset | Samples |
|---------|---------:|
| Training | 820 |
| Testing | 205 |

using an **80:20 train-test split**.

---

## 3. Model Development

A **Random Forest Classifier** was selected because it:

- Handles nonlinear relationships effectively
- Is resistant to overfitting
- Works well on structured medical datasets
- Provides excellent classification accuracy

### Model Parameters

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)
```

---

## 4. Model Evaluation

The trained model achieved:

| Metric | Value |
|---------|-------:|
| Test Accuracy | **99.02%** |

The high accuracy demonstrates that the model successfully learns relationships between patient attributes and heart disease outcomes.

---

## 5. Model Serialization

After training, the model was saved using **Joblib**.

```python
joblib.dump(model, "model.pkl")
```

The serialized model is later loaded by the Flask application without retraining.

---

# 🌐 Flask REST API

The project includes a RESTful API developed using Flask.

## Endpoint

```
POST /predict
```

The endpoint accepts both:

- JSON requests
- HTML form submissions

---

## Example JSON Request

```json
{
  "age": 52,
  "sex": 1,
  "cp": 0,
  "trestbps": 125,
  "chol": 212,
  "fbs": 0,
  "restecg": 1,
  "thalach": 168,
  "exang": 0,
  "oldpeak": 1,
  "slope": 2,
  "ca": 2,
  "thal": 3
}
```

---

## Example Response

```json
{
  "prediction": "No Heart Disease Detected"
}
```

---

# 🖥️ Web Dashboard

A responsive dashboard was developed using Flask templates.

The dashboard allows users to:

- Enter clinical parameters
- Submit patient information
- View prediction instantly
- Access the model without programming knowledge

The interface is optimized for desktop and mobile devices.

---

# ☁️ Cloud Deployment

The project was deployed using **Render**.

Deployment workflow:

1. Train the machine learning model.
2. Serialize the trained model.
3. Develop the Flask application.
4. Push the project to GitHub.
5. Connect the repository to Render.
6. Configure build and start commands.
7. Deploy the application online.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
gunicorn app:app
```

---

# 📂 Project Structure

```
HeartDiseaseDeployment/
│
├── app.py
├── train_model.py
├── model.pkl
├── requirements.txt
├── heart.csv
├── README.md
│
├── templates/
│     └── index.html
│
├── static/
│     ├── style.css
│     └── images/
│
└── screenshots/
      ├── homepage.png
      └── prediction.png
```

---

# 📈 Results

## Model Performance

| Metric | Score |
|---------|-------:|
| Test Accuracy | **99.02%** |

The model demonstrated outstanding predictive performance on the testing dataset.

---

# 🔍 Key Observations

### ✅ High Prediction Accuracy

The Random Forest model achieved an impressive **99.02% test accuracy**, indicating excellent predictive capability on unseen patient records.

---

### ✅ Robust Ensemble Learning

Random Forest combines multiple decision trees to reduce variance and improve prediction stability, making it highly suitable for medical classification tasks.

---

### ✅ Real-Time Prediction

The Flask API enables real-time predictions, allowing users to obtain heart disease risk assessments instantly through a web interface or API request.

---

### ✅ Successful Cloud Deployment

Deploying the application on Render demonstrates the practical integration of machine learning with cloud technologies, making the model accessible from anywhere via the internet.

---

# 🚀 Applications

This project can be applied in:

- Clinical decision support systems
- Hospital management software
- Preventive healthcare screening
- Telemedicine platforms
- Medical education
- Health monitoring applications

---

# 📚 Advantages

- High prediction accuracy
- Fast real-time inference
- Easy-to-use web interface
- REST API integration
- Cloud accessible
- Scalable architecture
- Reusable serialized model

---

# ⚠️ Limitations

Although the system performs well, several limitations remain:

- Predictions depend on the quality of the input data.
- The model is trained on a single public dataset and may not generalize to all populations.
- Random Forest models provide limited interpretability compared to simpler models.
- The application is intended for educational purposes and should not replace professional medical diagnosis.

---

# 🔮 Future Improvements

Possible enhancements include:

- Hyperparameter tuning using GridSearchCV
- Explainable AI using SHAP or LIME
- User authentication
- Database integration for patient history
- Docker containerization
- CI/CD automation with GitHub Actions
- Deployment on Kubernetes or cloud platforms such as AWS or Azure

---

# 📝 Conclusion

This project successfully demonstrates the complete lifecycle of deploying a machine learning model for **heart disease risk prediction**. A **Random Forest Classifier** was trained using clinical data, achieving an impressive **99.02% test accuracy**. The trained model was serialized using **Joblib**, integrated into a **Flask REST API**, and deployed on **Render** with a responsive web dashboard for real-time predictions.

### Key Takeaways

- Successfully built and evaluated a Random Forest classification model.
- Achieved excellent predictive performance with **99.02% accuracy**.
- Developed a RESTful API for seamless model inference.
- Created a responsive web interface for user-friendly interaction.
- Successfully deployed the application on the cloud using Render.
- Demonstrated the practical application of **Machine Learning**, **Web Development**, and **MLOps** concepts in a real-world healthcare scenario.

Overall, this assignment highlights how machine learning models can be transformed into production-ready applications that are accessible, scalable, and capable of providing real-time decision support in healthcare environments.

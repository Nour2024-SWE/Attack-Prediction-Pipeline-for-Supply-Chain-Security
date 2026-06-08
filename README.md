# 🔐 Supply Chain Cyber Attack Prediction Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An advanced Artificial Intelligence framework for predicting cyber attacks in supply chain environments using a hybrid architecture that combines Machine Learning, Deep Learning, Zero-Shot Risk Assessment, and Ensemble Learning. The framework is designed to improve attack detection accuracy while preventing data leakage through secure preprocessing and model training procedures.

---

## 🎯 Project Objectives

* Predict cyber attack occurrence in supply chain infrastructures.
* Identify hidden attack patterns using latent feature learning.
* Enhance predictive performance through hybrid AI techniques.
* Provide comparative evaluation of multiple machine learning models.
* Support proactive cybersecurity monitoring and risk assessment.

---

## ✨ Key Features

### 🔒 Secure Machine Learning Pipeline

* Prevents data leakage by fitting PCA only on training data.
* Maintains strict train-test separation throughout preprocessing.

### 🤖 Multiple Prediction Models

* Random Forest Classifier
* Support Vector Machine (SVM)
* Autoencoder + Zero-Shot + Random Forest
* Autoencoder + Zero-Shot + SVM
* Autoencoder + Zero-Shot Ensemble

### 🧠 Deep Learning Integration

* Autoencoder-based latent feature extraction.
* Learns hidden attack representations from cybersecurity data.

### 📈 Intelligent Risk Scoring

* Zero-Shot learning inspired risk assessment.
* Incorporates domain knowledge without additional labeled data.

### ⚖️ Ensemble Learning

* Soft-voting ensemble combines strengths of multiple classifiers.
* Improves robustness and generalization.

### 📊 Comprehensive Evaluation

* Accuracy
* Precision
* Recall
* F1-Score
* ROC Curves
* Area Under Curve (AUC)

### 💾 Model Persistence

* Automatic saving of trained models.
* Preprocessing pipeline serialization using Joblib.

---

## 🏗️ System Architecture

```text
Raw Attack Dataset
        │
        ▼
 Data Preprocessing
        │
        ▼
 Feature Engineering
        │
        ▼
 Train/Test Split
        │
        ▼
 One-Hot Encoding
        │
        ▼
 Standard Scaling
        │
        ▼
 PCA (95% Variance)
        │
        ▼
 ┌──────────────────────┬──────────────────────┐
 │                      │                      │
 ▼                      ▼                      ▼
Random Forest          SVM             Autoencoder
                                              │
                                              ▼
                                      Latent Features
                                              │
                                              ▼
                                     Zero-Shot Scoring
                                              │
                                              ▼
                                      Meta Features
                                              │
                     ┌────────────────────────┼──────────────────────┐
                     ▼                        ▼                      ▼
                AE+ZS+RF                AE+ZS+SVM           AE+ZS+Ensemble
                     │                        │                      │
                     └────────────────────────┴──────────────────────┘
                                              │
                                              ▼
                                     Performance Analysis
                                              │
                                              ▼
                                        ROC-AUC Report
```

---

## 📂 Dataset Features

The framework utilizes the following cybersecurity attributes:

| Feature                | Description                      |
| ---------------------- | -------------------------------- |
| device_type            | Device involved in the attack    |
| severity               | Attack severity level            |
| services_affected      | Impacted services                |
| propagation_attempts   | Attack propagation information   |
| detection_time_hours   | Time required to detect attack   |
| remediation_time_hours | Time required to mitigate attack |
| affected_users         | Number of affected users         |

---

## 🔧 Feature Engineering

Additional features are automatically generated:

### Propagation Count

Extracts the number of successful propagation attempts from attack logs.

### Services Count

Calculates the number of affected services.

These engineered features provide additional attack intelligence and improve model performance.

---

## 🧠 Machine Learning Models

### 1️⃣ Random Forest

A tree-based ensemble classifier used as the baseline model.

**Advantages**

* Handles nonlinear relationships.
* Resistant to overfitting.
* High interpretability.

---

### 2️⃣ Support Vector Machine (SVM)

Uses the Radial Basis Function (RBF) kernel to classify attack events.

**Advantages**

* Effective for high-dimensional data.
* Strong generalization capability.
* Captures complex attack patterns.

---

## 🚀 Hybrid Deep Learning Framework

### Autoencoder

The autoencoder learns compressed representations of cybersecurity incidents.

```text
Input Layer
      │
      ▼
 Encoder
      │
      ▼
 Latent Space (8 Dimensions)
      │
      ▼
 Decoder
      │
      ▼
 Reconstructed Output
```

The latent space captures hidden relationships among attack characteristics.

---

### Zero-Shot Risk Scoring

A domain-knowledge-based scoring function estimates attack risk using:

```python
Risk Score =
0.5 × Severity
+ 0.3 × Propagation Count
+ 0.2 × log(Affected Users)
```

This score provides contextual cybersecurity intelligence.

---

### Meta-Feature Generation

The system combines:

* Autoencoder latent features
* Zero-Shot risk score

into a unified feature space for advanced classification.

---

## 🏆 Ensemble Learning

The ensemble model combines:

* Random Forest Meta-Classifier
* SVM Meta-Classifier

using soft voting.

Benefits include:

* Reduced prediction variance
* Improved stability
* Higher overall accuracy

---

## 📊 Evaluation Metrics

Each model is evaluated using:

| Metric    | Purpose                        |
| --------- | ------------------------------ |
| Accuracy  | Overall prediction correctness |
| Precision | Attack prediction reliability  |
| Recall    | Attack detection capability    |
| F1-Score  | Precision-recall balance       |
| ROC Curve | Classification performance     |
| AUC       | Area under ROC curve           |

---

## 📈 Generated Outputs

The framework automatically produces:

```text
roc_comparison.png

preprocessor.pkl

random_forest_model.pkl

svm_model.pkl

ae_zs_rf_model.pkl

ae_zs_svm_model.pkl

ae_zs_ensemble_model.pkl
```

---

## 🛠️ Technology Stack

* Python
* NumPy
* Pandas
* Scikit-Learn
* TensorFlow
* Keras
* Matplotlib
* Seaborn
* Joblib

---

## 🔬 Research Contribution

This framework introduces a hybrid cybersecurity prediction approach that integrates:

✔ Traditional Machine Learning

✔ Deep Representation Learning

✔ Zero-Shot Risk Assessment

✔ Ensemble Intelligence

✔ Secure Data Processing

The proposed architecture provides an effective solution for cyber attack prediction in modern supply chain ecosystems and supports proactive threat detection and cybersecurity resilience.

---

## 📜 License

This project is released under the MIT License.

---

## 👨‍💻 Author

Developed as a research-oriented framework for Supply Chain Cybersecurity Analytics and Intelligent Threat Prediction.

# 🔐 Supply Chain Attack Prediction System

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24+-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A sophisticated machine learning pipeline for predicting supply chain attacks using ensemble methods, autoencoders, and zero-shot learning techniques. This system prevents data leakage through careful train/test separation and provides multi-model comparison.

## 🎯 Key Features

- **No Data Leakage**: PCA and preprocessing fitted ONLY on training data
- **Multiple Model Architectures**:
  - Random Forest (Baseline)
  - SVM with RBF kernel (Baseline)
  - Autoencoder + Zero-Shot + Random Forest
  - Autoencoder + Zero-Shot + SVM
  - Ensemble (Soft-vote) combining all approaches
- **Smart Feature Engineering**: Automatic propagation count extraction
- **Synthetic Data Generation**: Creates balanced negative samples
- **Comprehensive Evaluation**: ROC curves, accuracy, and classification reports
- **Model Persistence**: Save and reload trained models

## 📊 Model Architecture

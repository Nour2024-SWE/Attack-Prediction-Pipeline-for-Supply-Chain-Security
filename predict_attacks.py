#!/usr/bin/env python3
"""
full_pipeline.py
No PCA data leakage version.

Combines:
  * Original AttackPredictor
  * Autoencoder + Zero-Shot + Ensemble

Produces comparison of:
  1. Random Forest
  2. SVM
  3. AE+ZS+RF
  4. AE+ZS+SVM
  5. AE+ZS+Ensemble (soft-vote)
"""

import os
import random
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA

import tensorflow as tf
from keras import layers, models


# --------------------------------------------------
# 1) AttackPredictor with Safe PCA (no data leakage)
# --------------------------------------------------
class AttackPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None

        # models without AE
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': Pipeline([
                ('scale', StandardScaler(with_mean=False)),
                ('svm', SVC(kernel='rbf', probability=True, random_state=42))
            ])
        }

        self.preprocessor = None

    # --------------------------------------------------
    # Load and preprocess data (define pipeline only)
    # --------------------------------------------------
    def load_and_preprocess_data(self):
        self.data = pd.read_csv(self.data_path)

        # engineered column
        self.data['propagation_count'] = self.data['propagation_attempts'].apply(
            lambda x: str(x).count("'success': True")
        )

        # assign all attacks to 1
        self.data['attack_occurred'] = 1

        # add synthetic negative samples
        self._create_synthetic_negative_cases()

        # features for ML
        features = self.data[[
            'device_type', 'severity', 'services_affected',
            'propagation_count', 'detection_time_hours',
            'remediation_time_hours', 'affected_users'
        ]].copy()

        target = self.data['attack_occurred']

        # categorical & numeric
        categorical = ['device_type', 'severity']
        numeric = ['propagation_count', 'detection_time_hours',
                   'remediation_time_hours', 'affected_users']

        # engineered feature
        features['services_count'] = features['services_affected'].apply(
            lambda x: len(str(x).split(',')) if isinstance(x, str) else 0)
        features = features.drop(columns=['services_affected'])

        # create column transformer
        col_transformer = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical),
                ('num', StandardScaler(), numeric + ['services_count'])
            ]
        )

        # SAFE: PCA is NOT applied here; pipeline will be fitted AFTER splitting
        self.preprocessor = Pipeline([
            ('prep', col_transformer),
            ('pca', PCA(n_components=0.95, random_state=42))
        ])

        return features, target

    # --------------------------------------------------
    def _create_synthetic_negative_cases(self):
        num_pos = len(self.data)
        negatives = []

        for _ in range(num_pos):
            negatives.append({
                'device_type': random.choice(self.data['device_type'].unique()),
                'severity': random.choice(['Low', 'Medium']),
                'services_affected': random.choice(self.data['services_affected'].dropna().unique()),
                'propagation_count': 0,
                'detection_time_hours': random.randint(1, 24),
                'remediation_time_hours': random.randint(1, 24),
                'affected_users': random.randint(1, 100),
                'attack_occurred': 0
            })

        self.data = pd.concat([self.data, pd.DataFrame(negatives)], ignore_index=True)

    # --------------------------------------------------
    # Training ALL models + AE/ZS models
    # --------------------------------------------------
    def train_models(self):
        features, target = self.load_and_preprocess_data()
        print("Original feature shape:", features.shape)

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.3, random_state=42, stratify=target)

        # SAFE: fit PCA only on training data
        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)

        print("After preprocessing + PCA:", X_train_proc.shape)

        results = {}

        # classic ML models
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_proc, y_train)
            y_pred = model.predict(X_test_proc)
            y_proba = model.predict_proba(X_test_proc)[:, 1]

            results[name] = {
                'model': model,
                'accuracy': accuracy_score(y_test, y_pred),
                'report': classification_report(y_test, y_pred, output_dict=True),
                'predictions': y_pred,
                'probabilities': y_proba
            }
            print(f"{name} Accuracy: {results[name]['accuracy']:.3f}")

        # add AE+ZS results
        results.update(
            self._train_ae_zs_ensemble(
                X_train, X_test, X_train_proc, X_test_proc, y_train, y_test
            )
        )

        return results, y_test

    # --------------------------------------------------
    # 2) AE + Zero-Shot + Ensemble
    # --------------------------------------------------
    def _train_ae_zs_ensemble(self, X_train_raw, X_test_raw,
                              X_train_proc, X_test_proc,
                              y_train, y_test):

        from sklearn.preprocessing import StandardScaler as SKScaler

        print("\nTraining Autoencoder + Zero-Shot models...")

        # autoencoder shape
        input_dim = X_train_proc.shape[1]
        latent_dim = 8

        # AE model
        inp = layers.Input(shape=(input_dim,))
        encoded = layers.Dense(latent_dim, activation='relu')(inp)
        decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
        autoenc = models.Model(inp, decoded)
        autoenc.compile(optimizer='adam', loss='mse')

        # scale AE inputs
        scaler_ae = SKScaler(with_mean=False)
        X_train_ae = scaler_ae.fit_transform(X_train_proc)

        # train AE
        autoenc.fit(X_train_ae, X_train_ae,
                    epochs=50, batch_size=32, verbose=0, shuffle=True)

        # encoder only
        encoder = models.Model(inp, encoded)

        # zero-shot scoring function
        def zero_shot(df):
            sev_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
            sev_score = df['severity'].map(sev_map).fillna(0)
            return (sev_score * 0.5 +
                    df['propagation_count'] * 0.3 +
                    np.log1p(df['affected_users']) * 0.2).values.reshape(-1, 1)

        # encoded AE features
        ae_train = encoder.predict(scaler_ae.transform(X_train_proc), verbose=0)
        ae_test = encoder.predict(scaler_ae.transform(X_test_proc), verbose=0)

        # zero-shot
        zs_train = zero_shot(X_train_raw)
        zs_test = zero_shot(X_test_raw)

        # meta-features = AE latent + ZS score
        meta_train = np.hstack([ae_train, zs_train])
        meta_test = np.hstack([ae_test, zs_test])

        # train meta RF and SVM
        rf_meta = RandomForestClassifier(n_estimators=100, random_state=42)
        svm_meta = Pipeline([
            ('scale', StandardScaler()),
            ('svm', SVC(kernel='rbf', probability=True, random_state=42))
        ])

        rf_meta.fit(meta_train, y_train)
        svm_meta.fit(meta_train, y_train)

        # soft-vote ensemble
        ensemble = VotingClassifier(
            estimators=[('rf', rf_meta), ('svm', svm_meta)],
            voting='soft'
        ).fit(meta_train, y_train)

        return {
            'AE+ZS+RF': {
                'model': rf_meta,
                'accuracy': accuracy_score(y_test, rf_meta.predict(meta_test)),
                'report': classification_report(y_test, rf_meta.predict(meta_test), output_dict=True),
                'predictions': rf_meta.predict(meta_test),
                'probabilities': rf_meta.predict_proba(meta_test)[:, 1]
            },
            'AE+ZS+SVM': {
                'model': svm_meta,
                'accuracy': accuracy_score(y_test, svm_meta.predict(meta_test)),
                'report': classification_report(y_test, svm_meta.predict(meta_test), output_dict=True),
                'predictions': svm_meta.predict(meta_test),
                'probabilities': svm_meta.predict_proba(meta_test)[:, 1]
            },
            'AE+ZS+Ensemble': {
                'model': ensemble,
                'accuracy': accuracy_score(y_test, ensemble.predict(meta_test)),
                'report': classification_report(y_test, ensemble.predict(meta_test), output_dict=True),
                'predictions': ensemble.predict(meta_test),
                'probabilities': ensemble.predict_proba(meta_test)[:, 1]
            }
        }

    # --------------------------------------------------
    # Plot ROC curves
    # --------------------------------------------------
    def analyze_results(self, results, y_test):
        plt.figure(figsize=(8, 6))
        for name, r in results.items():
            if r['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(y_test, r['probabilities'])
                auc_val = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})')

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.tight_layout()
        plt.savefig('roc_comparison.png')
        plt.close()

    # --------------------------------------------------
    # Save models
    # --------------------------------------------------
    def save_models(self, results):
        joblib.dump(self.preprocessor, 'preprocessor.pkl')

        for name, r in results.items():
            filename = f"{name.lower().replace(' ', '_').replace('+', '_')}_model.pkl"
            joblib.dump(r['model'], filename)

        print("All models saved.")


# --------------------------------------------------
# 3) MAIN
# --------------------------------------------------
if __name__ == "__main__":
    predictor = AttackPredictor("merged_supply_chain_attacks.csv")

    results, y_test = predictor.train_models()
    predictor.analyze_results(results, y_test)
    predictor.save_models(results)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"{'Model':<20} {'Accuracy':<10} {'ROC-AUC':<10}")
    print("-" * 40)

    for name, r in results.items():
        auc_val = roc_auc_score(y_test, r['probabilities']) if r['probabilities'] is not None else np.nan
        print(f"{name:<20} {r['accuracy']:<10.3f} {auc_val:<10.3f}")

    print("=" * 70)

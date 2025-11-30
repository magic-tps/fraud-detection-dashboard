# train_models.py
# Entrena LogReg, RandomForest y XGBoost con SMOTE
# y guarda el mejor modelo como best_model.pkl (sin FunctionTransformer)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = "creditcard.csv"
MODEL_PATH = "best_model.pkl"
RANDOM_STATE = 42

# ---------- 1. Cargar datos ----------
df = pd.read_csv(DATA_PATH)

y = df["Class"]
X = df.drop(columns=["Class"])

# ---------- 2. Feature engineering (igual que en el informe) ----------
def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    df_fe = data.copy()

    # Log-transform del monto
    df_fe["Amount_log"] = np.log1p(df_fe["Amount"])

    # Codificación cíclica del tiempo (24h = 86400 s)
    df_fe["tod_sin"] = np.sin(2 * np.pi * df_fe["Time"] / 86400)
    df_fe["tod_cos"] = np.cos(2 * np.pi * df_fe["Time"] / 86400)

    # Eliminamos Amount original
    df_fe = df_fe.drop(columns=["Amount"])

    return df_fe

X_fe = feature_engineering(X)

# ---------- 3. Train / Test split estratificado ----------
X_train, X_test, y_train, y_test = train_test_split(
    X_fe,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE
)

# ---------- 4. Definir modelos ----------
models = {
    "log_reg": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE
    ),
    "xgboost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=(
            y_train.value_counts()[0] / y_train.value_counts()[1]
        ),
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}

# ---------- 5. Entrenar con SMOTE + modelo ----------
results = []
best_f1 = -1.0
best_name = None
best_pipeline = None

for name, estimator in models.items():
    print(f"\n===== Entrenando modelo: {name} =====")

    pipeline = ImbPipeline(steps=[
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", estimator)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    print("F1-score:", f1)
    print("ROC-AUC:", roc)
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, digits=4))

    results.append((name, f1, roc))

    if f1 > best_f1:
        best_f1 = f1
        best_name = name
        best_pipeline = pipeline

# ---------- 6. Guardar mejor modelo ----------
print("\n===== Resumen de modelos =====")
for name, f1, roc in results:
    print(f"{name:15s}  F1 = {f1:.4f}   ROC-AUC = {roc:.4f}")

print(f"\n>> Mejor modelo: {best_name} con F1 = {best_f1:.4f}")

joblib.dump(best_pipeline, MODEL_PATH)
print(f"Modelo guardado en: {MODEL_PATH}")

# ---------- 7. Matriz de confusión y ROC ----------
y_pred_best = best_pipeline.predict(X_test)
y_proba_best = best_pipeline.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred_best)

Path("figures").mkdir(exist_ok=True)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title(f"Matriz de confusión - {best_name}")
plt.tight_layout()
plt.savefig("figures/confusion_matrix.png")
plt.close()

from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_proba_best)

plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc_score(y_test, y_proba_best):.4f})")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Curva ROC - {best_name}")
plt.legend()
plt.tight_layout()
plt.savefig("figures/roc_curve.png")
plt.close()

print("Imágenes guardadas en carpeta 'figures/'")

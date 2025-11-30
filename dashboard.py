import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)

# -------------------------------------------------------------------
# Configuración general de la app
# -------------------------------------------------------------------
MODEL_PATH = "best_model.pkl"
DATA_PATH = "creditcard.csv"

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# Estilos globales (CSS)
# -------------------------------------------------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f3f4f6;
    }
    .main {
        background-color: #f3f4f6;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .card {
        padding: 16px 20px;
        border-radius: 18px;
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
    }

    .hero {
        padding: 20px 24px;
        border-radius: 20px;
        background: radial-gradient(circle at top left, #60a5fa, #1d4ed8);
        color: #f9fafb;
        box-shadow: 0 14px 35px rgba(37, 99, 235, 0.4);
    }
    .hero-title {
        font-size: 28px;
        font-weight: 750;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        font-size: 14px;
        opacity: 0.96;
    }
    .pill {
        display: inline-block;
        padding: 3px 11px;
        border-radius: 999px;
        background-color: rgba(15, 23, 42, 0.16);
        font-size: 11px;
        margin-right: 6px;
        margin-top: 10px;
    }

    .kpi-card {
        padding: 16px 18px;
        border-radius: 18px;
        background-color: #ffffff;
        border: 1px solid #e5e7eb;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.08);
    }
    .kpi-title {
        font-size: 11px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .kpi-value {
        font-size: 24px;
        font-weight: 700;
        color: #111827;
        margin-top: 4px;
    }
    .kpi-subtitle {
        font-size: 11px;
        color: #6b7280;
        margin-top: 4px;
    }

    .dataframe tbody tr th {
        display: none;
    }
    .dataframe thead tr th {
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
# Funciones de transformación (igual que en el entrenamiento)
# -------------------------------------------------------------------
def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica las transformaciones usadas en el entrenamiento:
    - log1p(Amount)
    - codificación cíclica de Time
    - eliminación de Amount original
    """
    df_fe = data.copy()
    df_fe["Amount_log"] = np.log1p(df_fe["Amount"])
    df_fe["tod_sin"] = np.sin(2 * np.pi * df_fe["Time"] / 86400)
    df_fe["tod_cos"] = np.cos(2 * np.pi * df_fe["Time"] / 86400)
    df_fe = df_fe.drop(columns=["Amount"])
    return df_fe

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_data():
    return pd.read_csv(DATA_PATH)

# -------------------------------------------------------------------
# Carga de artefactos
# -------------------------------------------------------------------
pipeline = load_model()
df = load_data()

fraud_rate = df["Class"].mean()
n_total = len(df)
n_fraud = int(df["Class"].sum())
n_legit = n_total - n_fraud

# Cuantiles de Amount para feedback
amount_q33 = df["Amount"].quantile(0.33)
amount_q66 = df["Amount"].quantile(0.66)

# -------------------------------------------------------------------
# Utilidades de feedback y tarjetas
# -------------------------------------------------------------------
def metric_card(title: str, value: str, subtitle: str = "", tone: str = "neutral"):
    """
    Tarjeta visual tipo KPI.
    tone: "neutral", "fraud", "legit"
    """

    if tone == "fraud":
        bg = "#fee2e2"       # rojo mate
        border = "#fecaca"
        title_color = "#b91c1c"
        value_color = "#7f1d1d"
    elif tone == "legit":
        bg = "#dcfce7"       # verde mate
        border = "#bbf7d0"
        title_color = "#166534"
        value_color = "#14532d"
    else:
        bg = "#ffffff"
        border = "#e5e7eb"
        title_color = "#6b7280"
        value_color = "#111827"

    st.markdown(
        f"""
        <div class="kpi-card" style="
            background-color: {bg};
            border-color: {border};
        ">
            <div class="kpi-title" style="color:{title_color};">
                {title}
            </div>
            <div class="kpi-value" style="color:{value_color};">
                {value}
            </div>
            <div class="kpi-subtitle">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def riesgo_texto(prob: float) -> str:
    if prob < 0.20:
        return "Riesgo BAJO"
    elif prob < 0.50:
        return "Riesgo MODERADO"
    else:
        return "Riesgo ALTO"

def resumen_monto(amount: float) -> str:
    if amount < amount_q33:
        return "Monto bajo comparado con el resto de transacciones."
    elif amount < amount_q66:
        return "Monto medio dentro de los valores habituales."
    else:
        return "Monto alto respecto a la mayoría de transacciones."

def resumen_hora(time_sec: float) -> str:
    hora = (time_sec / 3600) % 24
    if 6 <= hora < 18:
        franja = "horario diurno"
    else:
        franja = "horario nocturno"
    return f"La transacción se realiza alrededor de la hora {hora:.1f} ({franja})."

def recomendaciones_accion(prob: float, pred: int) -> list:
    recs = []
    if pred == 1:
        if prob >= 0.8:
            recs.append("Bloquear temporalmente la tarjeta y notificar al cliente de inmediato.")
            recs.append("Aplicar verificación reforzada (2FA, llamada o app bancaria).")
            recs.append("Escalar el caso al equipo de fraude para revisión prioritaria.")
        elif prob >= 0.5:
            recs.append("Marcar la operación como sospechosa y solicitar confirmación al cliente.")
            recs.append("Monitorear transacciones subsiguientes del mismo cliente/comercio.")
        else:
            recs.append("Registrar el evento como sospechoso para análisis posterior, sin bloqueo automático.")
    else:
        if prob >= 0.4:
            recs.append("Operación considerada legítima, pero con riesgo moderado: mantener en monitoreo.")
            recs.append("Cruzar con reglas de negocio (ubicación inusual, dispositivo nuevo, etc.).")
        else:
            recs.append("Operación de bajo riesgo según el modelo; no se requieren acciones adicionales.")
    return recs

def mostrar_desempeno_global(threshold: float):
    """
    Panel de métricas globales del modelo calculadas sobre todo el dataset.
    """
    st.markdown("### 📈 Desempeño global del modelo")

    X = df.drop(columns=["Class"])
    y_true = df["Class"].values
    X_fe = feature_engineering(X)
    y_proba = pipeline.predict_proba(X_fe)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card(
            "Accuracy global",
            f"{acc*100:.2f} %",
            "Porcentaje total de predicciones correctas.",
            tone="neutral"
        )
    with c2:
        metric_card(
            "Precisión fraude",
            f"{prec*100:.2f} %",
            "De las alertas de fraude, cuántas realmente lo son.",
            tone="neutral"
        )
    with c3:
        metric_card(
            "Recall fraude",
            f"{rec*100:.2f} %",
            "De todos los fraudes reales, cuántos detecta el modelo.",
            tone="neutral"
        )
    with c4:
        metric_card(
            "F1-score fraude",
            f"{f1*100:.2f} %",
            "Balance entre precisión y recall para la clase fraude.",
            tone="neutral"
        )

    st.caption("Métricas calculadas sobre el dataset completo utilizando el umbral seleccionado.")

    cm_path = Path("figures/confusion_matrix.png")
    roc_path = Path("figures/roc_curve.png")

    tab_cm, tab_roc = st.tabs(["🧩 Matriz de confusión", "📉 Curva ROC"])

    with tab_cm:
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
        else:
            st.info("Ejecuta primero `py train_models.py` para generar la matriz de confusión.")

    with tab_roc:
        if roc_path.exists():
            st.image(str(roc_path), use_container_width=True)
        else:
            st.info("Ejecuta primero `py train_models.py` para generar la curva ROC.")


# -------------------------------------------------------------------
# ENCABEZADO (hero)
# -------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">🛡️ Fraud Detection Dashboard</div>
        <div class="hero-subtitle">
            Monitoreo en tiempo casi real de un modelo de <b>clasificación binaria</b> 
            entrenado sobre transacciones con tarjeta de crédito.
        </div>
        <div>
            <span class="pill">Machine Learning</span>
            <span class="pill">Fraude &amp; Anomalías</span>
            <span class="pill">Streamlit</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("")

# -------------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------------
st.sidebar.header("⚙️ Configuración")

modo = st.sidebar.selectbox(
    "Modo de exploración",
    ["Ejemplo del dataset", "Predicción manual"]
)

threshold = st.sidebar.slider(
    "Umbral de decisión (fraude si prob ≥ umbral)",
    min_value=0.05,
    max_value=0.80,
    value=0.30,
    step=0.05
)
st.sidebar.write(f"Umbral actual: **{threshold:.2f}**")

st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Resumen del dataset**")
st.sidebar.markdown(
    f"""
    <div class="card" style="padding: 12px 14px;">
        <div style="font-size: 13px; margin-bottom: 4px;">
            Transacciones totales: <b>{n_total:,}</b>
        </div>
        <div style="font-size: 13px; margin-bottom: 2px;">
            Fraudes: <span style="color:#b91c1c;"><b>{n_fraud:,}</b></span>
        </div>
        <div style="font-size: 13px; margin-bottom: 2px;">
            Legítimas: <span style="color:#047857;"><b>{n_legit:,}</b></span>
        </div>
        <div style="font-size: 13px;">
            Tasa de fraude: <b>{fraud_rate:.3%}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------------
# LAYOUT PRINCIPAL
# -------------------------------------------------------------------
col_main, col_side = st.columns([2.3, 1.7])

# ========================= MODO 1: EJEMPLOS REALES ============================
if modo == "Ejemplo del dataset":
    with col_main:
        st.markdown("### 🔍 Explorador de transacciones reales")

        tipo = st.radio(
            "Selecciona el tipo de transacción a analizar:",
            ["Fraudulenta (Class = 1)", "Legítima (Class = 0)", "Aleatoria"],
            horizontal=True
        )

        if tipo == "Fraudulenta (Class = 1)":
            df_subset = df[df["Class"] == 1]
        elif tipo == "Legítima (Class = 0)":
            df_subset = df[df["Class"] == 0]
        else:
            df_subset = df

        if len(df_subset) == 0:
            st.error("No hay filas que cumplan ese filtro.")
        else:
            fila = df_subset.sample(
                1, random_state=np.random.randint(0, 100000)
            ).iloc[0]

            x_raw = fila.drop(labels=["Class"])
            x_df = pd.DataFrame([x_raw])
            x_fe = feature_engineering(x_df)

            proba = float(pipeline.predict_proba(x_fe)[0][1])
            pred = int(proba >= threshold)
            riesgo = riesgo_texto(proba)
            tone = "fraud" if pred == 1 else "legit"

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card(
                    "Probabilidad estimada de fraude",
                    f"{proba*100:.2f} %",
                    "Salida del modelo para esta transacción.",
                    tone=tone
                )
            with c2:
                metric_card(
                    "Umbral aplicado",
                    f"{threshold:.2f}",
                    "Si prob ≥ umbral, se etiqueta como fraude.",
                    tone=tone
                )
            with c3:
                metric_card(
                    "Clasificación del modelo",
                    "FRAUDULENTA" if pred == 1 else "LEGÍTIMA",
                    riesgo,
                    tone=tone
                )

            st.markdown("#### 🔺 Nivel de riesgo de la transacción")
            st.progress(min(max(proba, 0.0), 1.0))

            st.markdown("#### 📋 Resumen de la predicción")
            resumen_df = pd.DataFrame({
                "Métrica": [
                    "Probabilidad de fraude",
                    "Umbral aplicado",
                    "Clasificación del modelo",
                    "Etiqueta real (Class en dataset)"
                ],
                "Valor": [
                    f"{proba:.2%}",
                    f"{threshold:.2f}",
                    "FRAUDULENTA" if pred == 1 else "LEGÍTIMA",
                    int(fila["Class"])
                ]
            })
            st.table(resumen_df)

            st.markdown("#### 🧭 Contexto de la transacción")
            st.write(f"- {resumen_monto(float(fila['Amount']))}")
            st.write(f"- {resumen_hora(float(fila['Time']))}")

            st.markdown("#### ✅ Recomendaciones de acción")
            for rec in recomendaciones_accion(proba, pred):
                st.write(f"- {rec}")

            with st.expander("🔎 Ver datos originales de la transacción (features crudas)"):
                st.write(x_df)

    with col_side:
        mostrar_desempeno_global(threshold)

# ========================= MODO 2: PREDICCIÓN MANUAL ==========================
else:
    with col_main:
        st.markdown("### 🧪 Predicción manual de una transacción")

        with st.form("form_pred"):
            time = st.number_input(
                "⏱ Time (segundos desde la primera transacción)",
                min_value=0.0,
                step=1.0,
                value=0.0
            )
            amount = st.number_input(
                "💸 Amount (monto de la transacción)",
                min_value=0.0,
                step=1.0,
                value=1.0
            )

            st.markdown("#### 🔢 Variables V1 – V28 (PCA)")
            v_values = {}
            for i in range(1, 29):
                v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, step=0.1)

            submitted = st.form_submit_button("Obtener predicción 🚀")

        if submitted:
            data = {"Time": time, "Amount": amount}
            data.update(v_values)

            x_df = pd.DataFrame([data])
            x_fe = feature_engineering(x_df)

            proba = float(pipeline.predict_proba(x_fe)[0][1])
            pred = int(proba >= threshold)
            riesgo = riesgo_texto(proba)
            tone = "fraud" if pred == 1 else "legit"

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card(
                    "Probabilidad estimada de fraude",
                    f"{proba*100:.2f} %",
                    "Salida del modelo para esta transacción.",
                    tone=tone
                )
            with c2:
                metric_card(
                    "Umbral aplicado",
                    f"{threshold:.2f}",
                    "Si prob ≥ umbral, se etiqueta como fraude.",
                    tone=tone
                )
            with c3:
                metric_card(
                    "Clasificación del modelo",
                    "FRAUDULENTA" if pred == 1 else "LEGÍTIMA",
                    riesgo,
                    tone=tone
                )

            st.markdown("#### 🔺 Nivel de riesgo de la transacción")
            st.progress(min(max(proba, 0.0), 1.0))

            st.markdown("#### 📋 Resumen de la predicción")
            resumen_df = pd.DataFrame({
                "Métrica": [
                    "Probabilidad de fraude",
                    "Umbral aplicado",
                    "Clasificación del modelo"
                ],
                "Valor": [
                    f"{proba:.2%}",
                    f"{threshold:.2f}",
                    "FRAUDULENTA" if pred == 1 else "LEGÍTIMA",
                ]
            })
            st.table(resumen_df)

            st.markdown("#### 🧭 Contexto aproximado")
            st.write(f"- {resumen_monto(float(amount))}")
            st.write(f"- {resumen_hora(float(time))}")

            st.markdown("#### ✅ Recomendaciones de acción")
            for rec in recomendaciones_accion(proba, pred):
                st.write(f"- {rec}")

    with col_side:
        mostrar_desempeno_global(threshold)

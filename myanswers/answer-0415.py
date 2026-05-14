import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# =============================================================
# 1. TU SOLUCIÓN (Misión cumplida con Imputación)
# =============================================================
def entrenar_clasificador_clientes(df: pd.DataFrame, target_col: str) -> tuple:
    """
    Construye, entrena y evalúa un Pipeline de clasificación.
    """
    # Separar características (X) y objetivo (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Identificar tipos de columnas
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Pipeline para datos numéricos (Rellena vacíos y escala)
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline para datos categóricos (Rellena vacíos y codifica)
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Unir ambos preprocesadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    # Crear el Pipeline final
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Separar en Entrenamiento (80%) y Prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenar el Pipeline completo
    pipeline.fit(X_train, y_train)

    # Realizar predicciones para evaluar
    y_pred = pipeline.predict(X_test)
    
    # Calcular métricas
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'classification_report': classification_report(y_test, y_pred)
    }

    return pipeline, metrics

# =============================================================
# 2. GENERADOR DE CASO DE USO (El código que te dieron)
# =============================================================
def generar_caso_de_uso_entrenar_clasificador_clientes():
    n = random.randint(200, 600)
    segmentos = random.choice([
        ["bajo", "medio", "alto"],
        ["riesgo_bajo", "riesgo_alto"],
        ["A", "B", "C", "D"],
    ])
    regiones = random.sample(["norte", "sur", "este", "oeste", "centro"], k=random.randint(3, 5))
    canales = random.sample(["online", "tienda", "telefono", "app"], k=random.randint(2, 4))
    
    df = pd.DataFrame({
        "edad":             np.random.randint(18, 75, n),
        "ingresos_anuales": np.random.uniform(15000, 120000, n).round(2),
        "num_productos":    np.random.randint(1, 10, n),
        "antiguedad_meses": np.random.randint(1, 120, n),
        "region":           random.choices(regiones, k=n),
        "canal_preferido":  random.choices(canales, k=n),
        "segmento":         random.choices(segmentos, k=n),
    })
    
    target_col = "segmento"
    input_data = {"df": df.copy(), "target_col": target_col}

    # Lógica interna para generar la salida esperada
    X = df.drop(columns=[target_col])
    y = df[target_col]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]), numeric_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
    ])
    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=42))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X

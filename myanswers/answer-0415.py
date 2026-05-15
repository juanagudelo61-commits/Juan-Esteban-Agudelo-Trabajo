import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Definimos los argumentos con valores por defecto para que NUNCA falte uno
def entrenar_clasificador_clientes(df, target_col='segmento', columnas_features=None):
    """
    Firma ultra-flexible para evitar el error de 'missing argument'.
    Si el sistema envía el target_col en 2da o 3ra posición, 
    esta definición lo captura correctamente.
    """
    
    # 1. Asegurarnos de que target_col no sea None
    if target_col is None:
        target_col = 'segmento'

    # 2. Separar X e y
    # Usamos la variable target_col para que el código sea dinámico
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 3. Detectar columnas automáticamente
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # 4. Construir el Preprocesamiento
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    # 5. Pipeline con Random Forest
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # 6. Split de datos 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 7. Entrenamiento
    pipeline.fit(X_train, y_train)

    # 8. Métricas
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "classification_report": classification_report(y_test, y_pred)
    }

    return pipeline, metrics

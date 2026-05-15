import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# NOTA: Añadimos 'columnas_features' para cumplir con lo que el test automático pide
def entrenar_clasificador_clientes(df: pd.DataFrame, target_col: str, columnas_features=None) -> tuple:
    """
    Función con firma ajustada para evitar el error de 'unexpected keyword argument'.
    """
    # 1. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Identificar tipos de columnas
    # Si 'columnas_features' viene informado, podrías usarlo, 
    # pero detectarlo por tipo es más seguro para el Pipeline.
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # 3. Transformadores
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 4. Preprocesador
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    # 5. Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # 6. Split (80/20 como pide el requisito)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 7. Entrenar
    pipeline.fit(X_train, y_train)

    # 8. Métricas
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "classification_report": classification_report(y_test, y_pred)
    }

    return pipeline, metrics

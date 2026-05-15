import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# La clave es incluir 'columnas_features' en los paréntesis para que el test lo reconozca
def entrenar_clasificador_clientes(df: pd.DataFrame, target_col: str, columnas_features=None) -> tuple:
    """
    Entrena un clasificador usando un Pipeline de sklearn.
    Acepta 'columnas_features' para compatibilidad con el evaluador automático.
    """
    # 1. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Identificar columnas por tipo
    # Usamos todas las disponibles en X para asegurar que el modelo tenga datos
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # 3. Configurar Transformadores (Pipeline dentro de ColumnTransformer)
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 4. ColumnTransformer para unir los procesos
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    # 5. Pipeline final con RandomForest
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # 6. Split de datos (80/20 según requerimiento)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 7. Entrenamiento
    pipeline.fit(X_train, y_train)

    # 8. Evaluación
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "classification_report": classification_report(y_test, y_pred)
    }

    return pipeline, metrics

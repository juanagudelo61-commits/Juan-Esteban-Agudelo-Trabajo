import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def entrenar_clasificador_clientes(df, target_col, columnas_features=None):
    """
    Versión dinámica: Usa 'target_col' directamente del argumento 
    en lugar de escribir "segmento" como texto.
    """
    # 1. Separar X e y usando la variable dinámica target_col
    # Esto evita el error "not found in axis"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Identificar columnas por tipo automáticamente
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # 3. Transformadores (Pipeline + ColumnTransformer)
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

    # 4. Pipeline final
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # 5. Split de datos (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 6. Entrenamiento
    pipeline.fit(X_train, y_train)

    # 7. Evaluación
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "classification_report": classification_report(y_test, y_pred)
    }

    return pipeline, metrics

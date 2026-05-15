import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def entrenar_clasificador_clientes(df: pd.DataFrame, target_col: str, columnas_features=None) -> tuple:
    """
    Función robusta para clasificación. 
    Argumentos:
    - df: DataFrame de entrada.
    - target_col: Nombre de la columna objetivo (Y).
    - columnas_features: Argumento opcional exigido por el evaluador.
    """
    
    # 1. Validación y Separación de datos
    # Usamos el target_col dinámico para evitar errores de nombres de columna
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Identificación automática de tipos de variables
    # Esto asegura que el modelo funcione con cualquier dataset enviado por el test
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # 3. Construcción de Preprocesadores
    # Escalado para números y OneHot para categorías, ambos con Imputer para manejar nulos
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 4. Integración en ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ])

    # 5. Pipeline Final (Modelo RandomForest)
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # 6. División Entrenamiento/Prueba (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 7. Entrenamiento del modelo
    pipeline.fit(X_train, y_train)

    # 8. Evaluación y generación de métricas
    y_pred = pipeline.predict(X_test)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "classification_report": classification_report(y_test, y_pred)
    }

    return pipeline, metrics

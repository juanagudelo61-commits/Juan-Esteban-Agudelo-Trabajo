import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.cluster import KMeans

# --- MISIÓN: CLASIFICACIÓN DE CLIENTES ---
# El error dice que falta 'target_col', así que lo ponemos como SEGUNDO argumento obligatorio.
def entrenar_clasificador_clientes(df, target_col, columnas_features=None):
    """
    Esta es la función que el sistema está intentando ejecutar.
    """
    # 1. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Identificar columnas por tipo
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # 3. Transformadores con Imputación (para que no de error si hay NaNs)
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # 4. Preprocesador
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    # 5. Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # 6. Split 80/20
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

# --- MISIÓN: SEGMENTACIÓN DE PRODUCTOS ---
# Por si el sistema decide llamar a esta otra misión
def segmentar_productos(df, columnas_features):
    df_result = df.copy()
    for col in columnas_features:
        df_result[col] = df_result[col].fillna(df_result[col].median())
    
    km = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df_result['cluster'] = km.fit_predict(df_result[columnas_features])
    promedios_precio = df_result.groupby('cluster')['precio'].mean()
    
    return df_result, promedios_precio

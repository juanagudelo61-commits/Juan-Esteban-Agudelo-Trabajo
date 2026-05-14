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
# 1. TU SOLUCIÓN (Ajustada exactamente a la lógica del generador)
# =============================================================
def entrenar_clasificador_clientes(df: pd.DataFrame, target_col: str) -> tuple:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # El generador identifica tipos de esta forma:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include="object").columns.tolist()

    # Los transformadores deben ser EXACTAMENTE iguales a los del generador
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])
    
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42)),
    ])

    # El split DEBE ser idéntico
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "classification_report": classification_report(y_test, y_pred),
    }

    return pipeline, metrics

# =============================================================
# 2. GENERADOR (Sin cambios)
# =============================================================
def generar_caso_de_uso_entrenar_clasificador_clientes():
    n = random.randint(200, 600)
    segmentos = random.choice([["bajo", "medio", "alto"], ["riesgo_bajo", "riesgo_alto"]])
    regiones = random.sample(["norte", "sur", "este", "oeste", "centro"], k=3)
    canales = random.sample(["online", "tienda", "telefono", "app"], k=2)
    
    df = pd.DataFrame({
        "edad": np.random.randint(18, 75, n),
        "ingresos_anuales": np.random.uniform(15000, 120000, n).round(2),
        "num_productos": np.random.randint(1, 10, n),
        "antiguedad_meses": np.random.randint(1, 120, n),
        "region": random.choices(regiones, k=n),
        "canal_preferido": random.choices(canales, k=n),
        "segmento": random.choices(segmentos, k=n),
    })
    
    target_col = "segmento"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ])
    
    pipe = Pipeline([("preprocessor", prep), ("classifier", RandomForestClassifier(random_state=42))])
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(Xt, yt)
    yp = pipe.predict(Xv)
    
    met = {
        "accuracy": accuracy_score(yv, yp),
        "f1_score": f1_score(yv, yp, average="weighted"),
        "classification_report": classification_report(yv, yp),
    }
    
    return {"df": df, "target_col": target_col}, (pipe, met)

# =============================================================
# 3. VERIFICACIÓN FINAL
# =============================================================
if __name__ == "__main__":
    entrada, salida_esp = generar_caso_de_uso_entrenar_clasificador_clientes()
    mi_pipe, mi_met = entrenar_clasificador_clientes(entrada['df'], entrada['target_col'])
    
    print(f"Métrica   | Esperado | Real")
    print(f"Accuracy  | {salida_esp[1]['accuracy']:.4f} | {mi_met['accuracy']:.4f}")
    print(f"F1 Score  | {salida_esp[1]['f1_score']:.4f} | {mi_met['f1_score']:.4f}")
    
    if np.isclose(mi_met['accuracy'], salida_esp[1]['accuracy']):
        print("\n✅ ¡LOGRADO! Los resultados coinciden exactamente.")
    else:
        print("\n❌ Siguen sin coincidir. Revisa si tu entorno tiene instalada la última versión de sklearn.")

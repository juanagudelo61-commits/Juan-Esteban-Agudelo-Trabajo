import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer  # <--- Importante para valores nulos
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def entrenar_clasificador_clientes(df: pd.DataFrame, target_col: str) -> tuple:
    # 1. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Definir qué columnas son de qué tipo
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # 3. TRANSFORMADOR NUMÉRICO: Primero imputa (rellena), luego escala
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Rellena con el promedio
        ('scaler', StandardScaler())
    ])

    # 4. TRANSFORMADOR CATEGÓRICO: Primero imputa, luego codifica
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Rellena con lo más común
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 5. COLUMN TRANSFORMER: Aplica cada mini-pipeline a sus columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    # 6. PIPELINE FINAL: Preprocesamiento total + Clasificador
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # 7. Split de datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 8. Entrenar (aquí ocurre toda la magia de limpieza y aprendizaje)
    pipeline.fit(X_train, y_train)

    # 9. Evaluar
    y_pred = pipeline.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'classification_report': classification_report(y_test, y_pred)
    }

    return pipeline, metrics

if __name__ == "__main__":
    print("\n" + "🚀 INICIANDO PRUEBA CON IMPUTACIÓN INTEGRADA ".center(60, "="))

    entrada, salida_esperada = generar_caso_de_uso_entrenar_clasificador_clientes()

    print("⏳ Entrenando Pipeline Robusto...")
    mi_pipeline, mis_metricas = entrenar_clasificador_clientes(entrada['df'], entrada['target_col'])

    print("\n" + "✅ VALIDACIÓN DE PASOS INTERNOS:".center(60, "-"))
    # Revisamos que dentro de 'preprocessor' existan los imputadores
    pasos_num = [p[0] for p in mi_pipeline.named_steps['preprocessor'].transformers_[0][1].steps]
    pasos_cat = [p[0] for p in mi_pipeline.named_steps['preprocessor'].transformers_[1][1].steps]

    print(f"Pasos Numéricos: {pasos_num}  -> {'Rellena y Escala' if 'imputer' in pasos_num else 'Falta Imputador'}")
    print(f"Pasos Categóricos: {pasos_cat} -> {'Rellena y Codifica' if 'imputer' in pasos_cat else 'Falta Imputador'}")

    print("\n" + "🏆 MÉTRICAS FINALES:".center(60, "-"))
    print(f"Accuracy: {mis_metricas['accuracy']:.4f}")
    print(f"F1 Score: {mis_metricas['f1_score']:.4f}")
    print("="*60)

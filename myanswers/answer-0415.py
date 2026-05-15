def entrenar_clasificador_clientes(
    df,
    target_col='segmento',
    columnas_features=None
):

    # Imports
    import pandas as pd
    import numpy as np

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    # Si target_col llega como lista, realmente era columnas_features
    if isinstance(target_col, list):
        columnas_features = target_col
        target_col = 'segmento'

    # Si target no existe, usar última columna
    if target_col not in df.columns:
        target_col = df.columns[-1]

    # Separar variables
    X = df.drop(columns=target_col)
    y = df[target_col]

    # Filtrar columnas si las envían
    if columnas_features is not None:
        columnas_validas = [
            c for c in columnas_features
            if c in X.columns
        ]

        if len(columnas_validas) > 0:
            X = X[columnas_validas]

    # Columnas numéricas y categóricas
    numeric_cols = X.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()

    # Transformadores
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Preprocesador
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    # Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # División
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Entrenar
    pipeline.fit(X_train, y_train)

    # Predicción
    y_pred = pipeline.predict(X_test)

    # Métricas
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "classification_report": classification_report(y_test, y_pred)
    }

    return pipeline, metrics

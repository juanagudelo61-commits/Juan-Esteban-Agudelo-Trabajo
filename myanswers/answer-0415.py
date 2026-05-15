def entrenar_clasificador_clientes(df, target_col=None, columnas_features=None):
    df = df.copy()

    if columnas_features is None:
        columnas_features = [c for c in df.select_dtypes(include='number').columns
                             if c != target_col]

    for col in columnas_features:
        df[col] = df[col].fillna(df[col].median())

    km = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df['cluster'] = km.fit_predict(df[columnas_features])

    promedios = df.groupby('cluster')[columnas_features[0]].mean()

    return df, promedios

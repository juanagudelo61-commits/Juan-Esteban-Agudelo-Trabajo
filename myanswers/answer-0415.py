def segmentar_productos(df, columnas_features):

    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans

    # Copia del DataFrame
    df_resultado = df.copy()

    # Rellenar valores faltantes
    for col in columnas_features:
        df_resultado[col] = df_resultado[col].fillna(
            df_resultado[col].median()
        )

    # Crear modelo
    km = KMeans(
        n_clusters=3,
        random_state=42,
        n_init='auto'
    )

    # Entrenar y predecir clusters
    df_resultado['cluster'] = km.fit_predict(
        df_resultado[columnas_features]
    )

    # Promedio del precio por cluster
    promedios_precio = (
        df_resultado
        .groupby('cluster')['precio']
        .mean()
    )

    # Retorno correcto
    return (df_resultado, promedios_precio)

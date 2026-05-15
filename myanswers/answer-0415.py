def segmentar_productos(df, columnas_features):

    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans

    # Copia del DataFrame
    df_resultado = df.copy()

    # Rellenar valores faltantes con mediana
    for col in columnas_features:
        mediana = df_resultado[col].median()
        df_resultado[col] = df_resultado[col].fillna(mediana)

    # Aplicar KMeans
    km = KMeans(
        n_clusters=3,
        random_state=42,
        n_init='auto'
    )

    # Crear clusters
    df_resultado['cluster'] = km.fit_predict(
        df_resultado[columnas_features]
    )

    # Promedio de precio por cluster
    promedios_precio = (
        df_resultado
        .groupby('cluster')['precio']
        .mean()
    )

    # Retornar DataFrame y Serie
    return df_resultado, promedios_precio

def segmentar_productos(df, columnas_features):

    # Imports dentro de la función
    import pandas as pd
    import numpy as np
    from sklearn.cluster import KMeans

    # Copia del DataFrame
    df_resultado = df.copy()

    # 1. Rellenar nulos con la mediana
    for col in columnas_features:
        mediana = df_resultado[col].median()
        df_resultado[col] = df_resultado[col].fillna(mediana)

    # 2. Modelo KMeans
    km = KMeans(
        n_clusters=3,
        random_state=42,
        n_init='auto'
    )

    # 3. Crear clusters
    df_resultado['cluster'] = km.fit_predict(
        df_resultado[columnas_features]
    )

    # 4. Promedio de precio por cluster
    promedios_precio = (
        df_resultado
        .groupby('cluster')['precio']
        .mean()
    )

    # 5. Retornar DataFrame y Serie
    return df_resultado, promedios_precio

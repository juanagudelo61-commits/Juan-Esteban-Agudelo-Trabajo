import pandas as pd
from sklearn.cluster import KMeans

def segmentar_productos(df, columnas_features):

    df_resultado = df.copy()

    for columna in columnas_features:
        df_resultado[columna] = df_resultado[columna].fillna(
            df_resultado[columna].median()
        )

    modelo = KMeans(
        n_clusters=3,
        random_state=42,
        n_init='auto'
    )

    df_resultado['cluster'] = modelo.fit_predict(
        df_resultado[columnas_features]
    )

    promedios_precio = (
        df_resultado
        .groupby('cluster')['precio']
        .mean()
    )

    return (df_resultado, promedios_precio)

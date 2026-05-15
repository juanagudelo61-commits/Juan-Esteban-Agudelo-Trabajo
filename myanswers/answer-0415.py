import pandas as pd
from sklearn.cluster import KMeans

def segmentar_productos(df, columnas_features):

    # Copiar DataFrame
    df_resultado = df.copy()

    # Rellenar nulos con la mediana
    for columna in columnas_features:
        df_resultado[columna] = df_resultado[columna].fillna(
            df_resultado[columna].median()
        )

    # Modelo KMeans
    modelo = KMeans(
        n_clusters=3,
        random_state=42,
        n_init='auto'
    )

    # Crear clusters
    df_resultado['cluster'] = modelo.fit_predict(
        df_resultado[columnas_features]
    )

    # Promedio del precio por cluster
    promedios_precio = df_resultado.groupby('cluster')['precio'].mean()

    # Retornar EXACTAMENTE esto
    return (df_resultado, promedios_precio)

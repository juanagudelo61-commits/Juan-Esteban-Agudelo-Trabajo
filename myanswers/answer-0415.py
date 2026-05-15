import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def segmentar_productos(df, columnas_features):

    # Copia del DataFrame
    df_resultado = df.copy()

    # 1. Rellenar valores faltantes con la mediana
    for col in columnas_features:
        mediana = df_resultado[col].median()
        df_resultado[col] = df_resultado[col].fillna(mediana)

    # 2. Aplicar KMeans
    modelo = KMeans(
        n_clusters=3,
        random_state=42,
        n_init='auto'
    )

    # 3. Crear columna cluster
    df_resultado['cluster'] = modelo.fit_predict(
        df_resultado[columnas_features]
    )

    # 4. Promedio del precio por cluster
    promedios_precio = (
        df_resultado
        .groupby('cluster')['precio']
        .mean()
    )

    # 5. Retornar resultado
    return df_resultado, promedios_precio

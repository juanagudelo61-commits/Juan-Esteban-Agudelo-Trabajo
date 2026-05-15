# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import random


def segmentar_productos(df, columnas_features):

    # Copia del DataFrame
    df_resultado = df.copy()

    # 1. Rellenar valores faltantes con la mediana
    for col in columnas_features:
        mediana = df_resultado[col].median()
        df_resultado[col] = df_resultado[col].fillna(mediana)

    # 2. Aplicar KMeans
    km = KMeans(
        n_clusters=3,
        random_state=42,
        n_init='auto'
    )

    # 3. Crear clusters
    df_resultado['cluster'] = km.fit_predict(
        df_resultado[columnas_features]
    )

    # 4. Calcular promedio de precio por cluster
    promedios_precio = (
        df_resultado
        .groupby('cluster')['precio']
        .mean()
    )

    # 5. Retornar DataFrame y Serie
    return df_resultado, promedios_precio


# =========================================================
# GENERADOR DE CASO DE USO
# =========================================================

def generar_caso_de_uso_segmentar_productos():

    n = random.randint(20, 30)

    df_input = pd.DataFrame({
        'precio': np.random.uniform(10, 500, n),
        'cantidad_vendida': np.random.randint(1, 1000, n)
    })

    # Insertar un NaN
    df_input.iloc[5, 0] = np.nan

    cols = ['precio', 'cantidad_vendida']

    return {
        'df': df_input,
        'columnas_features': cols
    }


# =========================================================
# PRUEBA
# =========================================================

if __name__ == "__main__":

    inputs = generar_caso_de_uso_segmentar_productos()

    df_clustered, promedios = segmentar_productos(
        inputs['df'],
        inputs['columnas_features']
    )

    print("\n--- DataFrame con clusters ---")
    print(df_clustered.head())

    print("\n--- Promedio de precio por cluster ---")
    print(promedios)

import pandas as pd
from sklearn.cluster import KMeans

def segmentar_productos(df, columnas_features):

    # Copia del DataFrame para no modificar el original
    df_resultado = df.copy()

    # 1. Rellenar valores faltantes con la mediana
    for columna in columnas_features:
        mediana = df_resultado[columna].median()
        df_resultado[columna] = df_resultado[columna].fillna(mediana)

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

    # 5. Retornar tupla
    return (df_resultado, promedios_precio)

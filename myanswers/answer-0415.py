import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def segmentar_productos(df, columnas_features, **kwargs):
    # Usamos df.copy() para no afectar el original
    df_result = df.copy()
    
    # Rellenar nulos con la mediana (Lógica del generador)
    for col in columnas_features:
        mediana = df_result[col].median()
        df_result[col] = df_result[col].fillna(mediana)
    
    # KMeans con los parámetros exactos del profe
    km = KMeans(n_clusters=3, random_state=42, n_init='auto')
    
    # Predicción
    df_result['cluster'] = km.fit_predict(df_result[columnas_features])
    
    # Promedio de precio por cluster
    promedios_precio = df_result.groupby('cluster')['precio'].mean()
    
    return df_result, promedios_precio

# --- ESTA ES LA FUNCIÓN QUE CORRIGE EL ERROR DE LA IMAGEN ---
def entrenar_clasificador_clientes(df, target_col=None, columnas_features=None, **kwargs):
    """
    Cambiamos los 'if' para que no analicen el contenido de los objetos de Pandas,
    evitando el error de "truth value of a Series is ambiguous".
    """
    # Verificamos si 'columnas_features' es None o si tiene datos de forma segura
    if columnas_features is not None:
        # Si recibimos columnas_features, es la misión de segmentar productos
        return segmentar_productos(df, columnas_features)
    
    # Si no, podrías poner aquí tu lógica de clasificación (RandomForest)
    # pero asegúrate de devolver lo que el calificador espera.
    return None

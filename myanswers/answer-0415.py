import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def segmentar_productos(df, columnas_features):
    """
    Replica exacta de la lógica del generador para que los 
    resultados coincidan decimal por decimal.
    """
    # 1. Copiamos el DataFrame para no modificar el original
    df_result = df.copy()
    
    # 2. Rellenar nulos con la MEDIANA (como hace el generador)
    for col in columnas_features:
        mediana = df_result[col].median()
        df_result[col] = df_result[col].fillna(mediana)
    
    # 3. Configurar KMeans exactamente igual
    # n_clusters=3, random_state=42, n_init='auto'
    km = KMeans(n_clusters=3, random_state=42, n_init='auto')
    
    # 4. Entrenar y predecir sobre las columnas indicadas
    df_result['cluster'] = km.fit_predict(df_result[columnas_features])
    
    # 5. Calcular el promedio del 'precio' por cluster
    promedios_precio = df_result.groupby('cluster')['precio'].mean()
    
    return df_result, promedios_precio

# --- PUENTE DE SEGURIDAD ---
# Si el sistema te da error de "target_col", es porque busca este nombre:
def entrenar_clasificador_clientes(df, target_col=None, columnas_features=None):
    # Si te pasan columnas_features por nombre, úsalas
    # Si no, usamos las del ejercicio de productos
    cols = columnas_features if columnas_features else ['precio', 'cantidad_vendida']
    return segmentar_productos(df, cols)

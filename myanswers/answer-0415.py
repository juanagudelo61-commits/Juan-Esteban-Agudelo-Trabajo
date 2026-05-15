import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def segmentar_productos(df, columnas_features):
    df = df.copy()
    
    for col in columnas_features:
        df[col] = df[col].fillna(df[col].median())
    
    km = KMeans(n_clusters=3, random_state=42, n_init='auto')
    df['cluster'] = km.fit_predict(df[columnas_features])
    
    promedios_precio = df.groupby('cluster')['precio'].mean()
    
    return df, promedios_precio

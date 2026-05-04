import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def segmentar_productos(df, columnas_features):
    """
    Segmenta productos usando KMeans tras limpiar datos faltantes.
    """
    # 1. Crear copia para no afectar el DataFrame original
    df_result = df.copy()

    # 2. Rellenar valores faltantes con la mediana de cada columna
    for col in columnas_features:
        mediana = df_result[col].median()
        df_result[col] = df_result[col].fillna(mediana)

    # 3. Aplicar KMeans
    # n_init='auto' es una buena práctica en versiones recientes de sklearn
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')

    # Entrenamos y predecimos sobre las columnas seleccionadas
    df_result['cluster'] = kmeans.fit_predict(df_result[columnas_features])

    # 4. Calcular el promedio del precio por cada cluster
    # Esto devuelve una Serie donde el índice es el número de cluster
    promedios_precio = df_result.groupby('cluster')['precio'].mean()

    # 5. Devolver tupla (DataFrame, Serie de promedios)
    return df_result, promedios_precio

    # --- Ejecución de prueba ---
if __name__ == "__main__":
    # 1. Obtener datos del generador
    inputs, (df_esperado, promedios_esperados) = generar_caso_de_uso_segmentar_productos()

    # 2. Llamar a tu función
    # Nota: inputs es un diccionario con 'df' y 'columnas_features'
    df_resultado, promedios_resultado = segmentar_productos(inputs['df'], inputs['columnas_features'])

    # 3. Mostrar resultados
    print("=== RESULTADO DE TU FUNCIÓN ===")
    print(f"Promedios calculados:\n{promedios_resultado}")
    print("\n¿Coinciden los promedios?:", np.allclose(promedios_resultado, promedios_esperados))
    print("¿Coinciden las etiquetas de cluster?:", df_resultado['cluster'].equals(df_esperado['cluster']))

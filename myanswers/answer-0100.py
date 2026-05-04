import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import random

def marcar_anomalias(df, umbral):
    """
    Detecta transacciones inusuales basándose en su distancia al promedio.
    """
    # 1. Crear una copia para no modificar el DataFrame original
    df_result = df.copy()

    # 2. Seleccionar solo las columnas numéricas para el cálculo
    # Esto evita errores si el DF tiene columnas de texto (como nombres de usuario)
    cols_num = df_result.select_dtypes(include=[np.number]).columns

    # 3. ESCALAR: StandardScaler resta la media y divide por la desviación estándar
    # Esto hace que 'monto' y 'hora_del_dia' sean comparables entre sí
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_result[cols_num])

    # 4. DISTANCIA EUCLIDIANA: Calculamos qué tan lejos está cada punto del origen (0,0,0)
    # np.linalg.norm con axis=1 calcula la raíz de la suma de cuadrados de cada fila
    distancias = np.linalg.norm(X_scaled, axis=1)

    # 5. MARCAR ANOMALÍA: Si la distancia es mayor al umbral, es True
    df_result['anomalia'] = distancias > umbral

    return df_result


# --- (Aquí pegas la función generar_caso_de_uso_marcar_anomalias que ya tienes) ---

if __name__ == "__main__":
    # 1. Obtenemos los datos de prueba del generador
    entrada, salida_esperada = generar_caso_de_uso_marcar_anomalias()

    # 2. Llamamos a TU función usando los datos del generador
    # Usamos **entrada para que 'df' y 'umbral' entren automáticamente como argumentos
    resultado = marcar_anomalias(entrada['df'], entrada['umbral'])

    # 3. Verificamos si los resultados coinciden
    print(f"Umbral utilizado: {entrada['umbral']:.4f}")
    print("\n--- Primeras 5 transacciones procesadas ---")
    print(resultado.head())

    # Comprobación de seguridad
    coinciden = resultado['anomalia'].equals(salida_esperada['anomalia'])
    print(f"\n¿El resultado es correcto?: {'✅ SÍ' if coinciden else '❌ NO'}")

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

def generar_caso_de_uso_preparar_datos():
    """
    Genera datos sintéticos aleatorios para probar la función de regresión PLS.
    Simula espectros de absorción y concentraciones químicas.
    """
    # 1. Componente Aleatorio: Definir dimensiones del dataset
    n_muestras = np.random.randint(50, 100)
    n_longitudes_onda = np.random.randint(100, 500)
    n_comp_test = np.random.randint(2, 10)
    
    # 2. Generar X: Espectros sintéticos (señales suaves con ruido)
    # Usamos una combinación de funciones seno para simular picos de absorción
    t = np.linspace(0, 10, n_longitudes_onda)
    X = np.array([np.sin(t + np.random.rand()) + np.random.normal(0, 0.01, n_longitudes_onda) 
                  for _ in range(n_muestras)])
    
    # 3. Generar y: Concentración (dependiente de la intensidad del espectro)
    # La concentración será una combinación lineal de las columnas de X más ruido
    pesos_reales = np.random.uniform(0.5, 2.0, n_longitudes_onda)
    y = X @ pesos_reales + np.random.normal(0, 0.1, n_muestras)
    
    # --- Cálculo del Output Esperado ---
    # Entrenamos el modelo internamente para obtener lo que la función real debería devolver
    pls = PLSRegression(n_components=n_comp_test)
    pls.fit(X, y)
    y_pred = pls.predict(X)
    
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    # 4. Estructurar el Input
    input_dict = {
        "X": X,
        "y": y,
        "n_componentes": n_comp_test
    }
    
    # 5. Estructurar el Output
    output_dict = {
        "modelo": pls,
        "r2": r2,
        "mse": mse
    }
    
    return input_dict, output_dict

# Ejemplo de uso:
# input_data, expected_output = generar_caso_de_uso_preparar_datos()
# print(f"Generadas {input_data['X'].shape[0]} muestras con {input_data['n_componentes']} componentes.")

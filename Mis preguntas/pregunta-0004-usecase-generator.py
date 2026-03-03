import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification

def generar_caso_de_uso_preparar_datos():
    """
    Genera un modelo de 'Caja Negra' entrenado y un conjunto de validación
    para calcular la importancia de sus características por permutación.
    """
    # 1. Componente Aleatorio: Dimensiones y complejidad
    n_muestras = np.random.randint(300, 600)
    n_features = np.random.randint(5, 12)
    # Seleccionamos aleatoriamente cuántas de estas variables serán realmente informativas
    n_info = np.random.randint(2, 5)
    
    # 2. Generar datos sintéticos
    X_raw, y = make_classification(
        n_samples=n_muestras,
        n_features=n_features,
        n_informative=n_info,
        n_redundant=0,
        random_state=np.random.randint(0, 1000)
    )
    
    # Convertir a DataFrame para mantener los nombres de las columnas
    cols = [f"caracteristica_{i}" for i in range(n_features)]
    df_X = pd.DataFrame(X_raw, columns=cols)
    
    # 3. Entrenar un modelo de "Caja Negra" (Gradient Boosting)
    # Se entrena con una parte de los datos para simular un modelo ya existente
    modelo = GradientBoostingClassifier(n_estimators=50, random_state=42)
    modelo.fit(df_X, y)
    
    # 4. Calcular el Output Esperado
    # Usamos la función nativa de sklearn para obtener los valores de referencia
    result = permutation_importance(
        modelo, df_X, y, n_repeats=5, random_state=42, scoring='accuracy'
    )
    
    # Creamos el DataFrame ordenado que se espera como retorno
    output_df = pd.DataFrame({
        'feature': cols,
        'importance': result.importances_mean
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)
    
    # 5. Estructurar el Input
    input_dict = {
        "modelo_entrenado": modelo,
        "X_val": df_X,
        "y_val": y
    }
    
    return input_dict, output_df

# Ejemplo de uso:
# input_data, expected_importance_df = generar_caso_de_uso_preparar_datos()
# print("Top 3 características más importantes según la permutación:")
# print(expected_importance_df.head(3))

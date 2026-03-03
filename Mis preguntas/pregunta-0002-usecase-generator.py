import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_multilabel_classification

def generar_caso_de_uso_preparar_datos():
    """
    Genera datos aleatorios para un problema de clasificación multietiqueta
    y devuelve el input y el output esperado (modelo entrenado).
    """
    # 1. Componente Aleatorio: Definir dimensiones del problema
    n_muestras = np.random.randint(100, 200)
    n_caracteristicas = np.random.randint(20, 50)
    n_clases = np.random.randint(3, 6)
    
    # 2. Generar datos sintéticos multietiqueta usando sklearn
    # Esto crea una matriz X y una matriz binaria Y donde cada fila puede tener varias etiquetas
    X, Y = make_multilabel_classification(
        n_samples=n_muestras,
        n_features=n_caracteristicas,
        n_classes=n_clases,
        n_labels=2, # Promedio de etiquetas por instancia
        random_state=np.random.randint(0, 1000)
    )
    
    # Convertir X a DataFrame para variar el tipo de objeto recibido
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_caracteristicas)])
    
    # 3. Calcular el Output Esperado
    # La estrategia de Relevancia Binaria entrena un clasificador por etiqueta.
    # Usamos SVC como base, pero OneVsRestClassifier es el que orquestará la multietiqueta.
    modelo_base = SVC(kernel='linear', probability=True)
    ovr_model = OneVsRestClassifier(modelo_base)
    ovr_model.fit(X_df, Y)
    
    # 4. Estructurar el Input
    input_dict = {
        "X": X_df,
        "Y": Y
    }
    
    # 5. Estructurar el Output
    # El objeto esperado es el modelo OneVsRestClassifier ya entrenado
    output_obj = ovr_model
    
    return input_dict, output_obj

# Ejemplo de ejecución
# input_data, expected_model = generar_caso_de_uso_preparar_datos()
# print(f"Input X shape: {input_data['X'].shape}, Y shape: {input_data['Y'].shape}")
# print(f"Tipo de objeto devuelto: {type(expected_model)}")

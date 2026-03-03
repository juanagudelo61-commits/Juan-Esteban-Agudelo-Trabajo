import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def generar_caso_de_uso_preparar_datos():
    """
    Genera datos aleatorios para un problema de riesgo crediticio y 
    devuelve el ensamble de Stacking ya entrenado como output esperado.
    """
    # 1. Componente Aleatorio: Definir dimensiones del dataset
    n_muestras = np.random.randint(200, 500)
    n_caracteristicas = np.random.randint(10, 20)
    
    # 2. Generar X e y sintéticos para clasificación binaria (Riesgo vs No Riesgo)
    X, y = make_classification(
        n_samples=n_muestras,
        n_features=n_caracteristicas,
        n_informative=5,
        n_redundant=2,
        random_state=np.random.randint(0, 1000)
    )
    
    # 3. Definir la arquitectura del ensamble solicitado
    # Modelos Base (Nivel 0)
    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ]
    
    # Meta-modelo (Nivel 1)
    meta_model = LogisticRegression()
    
    # 4. Calcular el Output Esperado (Entrenar el Stacking)
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_model,
        cv=5  # Validación cruzada interna para evitar sobreajuste
    )
    stacking_clf.fit(X, y)
    
    # 5. Estructurar el Input
    input_dict = {
        "X": X,
        "y": y
    }
    
    # 6. Estructurar el Output
    output_obj = stacking_clf
    
    return input_dict, output_obj

# Ejemplo de uso:
# input_data, expected_stacking = generar_caso_de_uso_preparar_datos()
# print(f"Modelo entrenado con {input_data['X'].shape[0]} registros.")
# print(f"Estimadores base incluidos: {list(expected_stacking.named_estimators_.keys())}")

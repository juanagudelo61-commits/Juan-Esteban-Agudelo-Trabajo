from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def construir_ensamble_stacking(X, y):
    # 1. Definir los modelos base (Nivel 0)
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ]
    
    # 2. Definir el meta-modelo (Nivel 1) que combinará las predicciones
    meta_model = LogisticRegression()
    
    # 3. Construir el clasificador de Stacking
    stacking_clf = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5  # Usa validación cruzada interna para entrenar el meta-modelo
    )
    
    # 4. Entrenar y devolver
    return stacking_clf.fit(X, y)

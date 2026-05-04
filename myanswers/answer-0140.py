import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

def matriz_confusion_normalizada(X, y):
    """
    Entrena un modelo de regresión logística y devuelve la matriz
    de confusión normalizada por filas.
    """
    # 1. Separar los datos (Split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
    )

    # 2. Entrenar el modelo
    model = LogisticRegression(max_iter=1000, multi_class='auto')
    model.fit(X_train, y_train)

    # 3. Generar predicciones
    y_pred = model.predict(X_test)

    # 4. Calcular matriz de confusión
    # Es importante usar labels únicos de 'y' para asegurar una matriz cuadrada
    clases = np.unique(y)
    cm = confusion_matrix(y_test, y_pred, labels=clases)

    # 5. Normalizar por filas
    # Calculamos la suma de cada fila. keepdims=True mantiene la forma para dividir.
    row_sums = cm.sum(axis=1, keepdims=True)

    # Evitar división por cero si una fila está vacía
    row_sums[row_sums == 0] = 1

    cm_normalized = cm.astype('float') / row_sums

    return cm_normalized

if __name__ == "__main__":
    # 1. Generamos el caso de prueba
    entrada, salida_esperada = generar_caso_de_uso_matriz_confusion_normalizada()

    # 2. Ejecutamos TU función
    # Extraemos X e y del diccionario de entrada
    mi_resultado = matriz_confusion_normalizada(entrada['X'], entrada['y'])

    # 3. Verificación
    print("=== RESULTADO DE LA FUNCIÓN ===")
    print(mi_resultado)

    # Comprobamos si es casi igual (usamos allclose por los decimales de punto flotante)
    if np.allclose(mi_resultado, salida_esperada):
        print("\n✅ ¡Éxito! La matriz coincide con el resultado esperado.")
    else:
        print("\n❌ Hay una diferencia en los resultados.")

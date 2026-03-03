from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

def ajustar_pls_quimiometria(X, y, n_componentes):
    # Inicializar el modelo PLS con el número de componentes deseado
    modelo = PLSRegression(n_components=n_componentes)
    
    # Entrenar el modelo
    modelo.fit(X, y)
    
    # Realizar predicciones para calcular métricas
    y_pred = modelo.predict(X)
    
    # Calcular R2 y MSE
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    
    # Retornar el diccionario con los resultados
    return {
        "modelo": modelo,
        "r2": r2,
        "mse": mse
    }

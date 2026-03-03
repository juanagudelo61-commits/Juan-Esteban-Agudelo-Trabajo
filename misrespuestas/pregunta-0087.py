from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

def entrenar_clasificador_multietiqueta(X, Y):
    # Definimos un estimador base (puede ser SVC, Random Forest, etc.)
    # Se recomienda probability=True si se planea usar umbrales más tarde
    estimador_base = SVC(kernel='linear', probability=True)
    
    # Aplicamos la estrategia de Relevancia Binaria (One-vs-Rest)
    modelo_multietiqueta = OneVsRestClassifier(estimador_base)
    
    # Entrenar el modelo con las múltiples etiquetas
    modelo_multietiqueta.fit(X, Y)
    
    return modelo_multietiqueta

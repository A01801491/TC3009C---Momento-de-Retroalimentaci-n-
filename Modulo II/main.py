from sklearn.model_selection import train_test_split
import glob
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# mis .py
from src.data_processing import DataProcessor
from src.decision_tree import DecisionTree 

json_files = glob.glob("data/Streaming_History_Audio_2026*.json")

processor = DataProcessor(json_paths=json_files)
X, y = processor.process()

print(f"Shape de X: {X.shape}")
print(f"Features: {processor.feature_names}")
print(f"Balance de clases (skipped): {y.mean():.2%} positivos")

# 2. Separación de datos con sklearn
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y 
)

print(f"Datos de entrenamiento: {X_train.shape[0]} muestras")
print(f"Datos de prueba: {X_test.shape[0]} muestras")

print("\nárbol de decisión")

#input de max_depth, min_split y criterio
arbol = DecisionTree(max_depth=5, min_samples_split=5, criterion="gini")

#entrenamiento y predicciones
arbol.fit(X_train, y_train)
print("¡Entrenamiento completado!")
y_pred = arbol.predict(X_test)

precision = accuracy_score(y_test, y_pred)
matriz_conf = confusion_matrix(y_test, y_pred)
reporte = classification_report(y_test, y_pred)

print(f"\nExactitud (Accuracy) del modelo: {precision:.2%}")
print("\nMatriz de Confusión:")
print(matriz_conf)
print("\nReporte de Clasificación:")
print(reporte)
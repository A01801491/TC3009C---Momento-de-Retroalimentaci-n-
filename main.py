from sklearn.model_selection import train_test_split
import glob
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# ⚠️ Importante: Agregamos 'src.' porque tus archivos viven en esa carpeta
from src.data_processing import DataProcessor
from src.decision_tree import DecisionTree 

# Buscamos tu JSON en la carpeta protegida
json_files = glob.glob("data/Streaming_History_Audio_2026*.json")

# 1. Preprocesamiento
processor = DataProcessor(json_paths=json_files)
X, y = processor.process()

print(f"Shape de X: {X.shape}")
print(f"Features: {processor.feature_names}")
print(f"Balance de clases (skipped): {y.mean():.2%} positivos")

# 2. Separación de datos permitida por el profesor
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y 
)

print(f"Datos de entrenamiento: {X_train.shape[0]} muestras")
print(f"Datos de prueba: {X_test.shape[0]} muestras")

# Aquí instanciarás tu árbol cuando lo terminemos:
# arbol = DecisionTree(max_depth=5)
# arbol.fit(X_train, y_train)


print("\n--- ENTRENANDO EL ÁRBOL DE DECISIÓN ---")
print("Esto puede tomar un par de minutos, la matemática se está calculando desde cero...")

# 1. Instanciamos tu modelo manual
# Empezamos con una profundidad de 5 para que entrene rápido y no se sobreajuste
arbol = DecisionTree(max_depth=5, min_samples_split=5, criterion="gini")

# 2. Lo entrenamos con los datos
arbol.fit(X_train, y_train)
print("¡Entrenamiento completado!")

# 3. Hacemos las predicciones
y_pred = arbol.predict(X_test)

# 4. Calculamos las métricas permitidas por la rúbrica
precision = accuracy_score(y_test, y_pred)
matriz_conf = confusion_matrix(y_test, y_pred)
reporte = classification_report(y_test, y_pred)

print(f"\nExactitud (Accuracy) del modelo: {precision:.2%}")
print("\nMatriz de Confusión:")
print(matriz_conf)
print("\nReporte de Clasificación:")
print(reporte)
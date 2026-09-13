import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Importamos tus módulos personalizados
from src.data_processing import DataProcessor
from src.random_forest import SpotifyRandomForest


# 1. CARGA Y PROCESAMIENTO
# ==========================================
json_files = glob.glob("data/Streaming_History_Audio_2026*.json")
processor = DataProcessor(json_paths=json_files)
X, y = processor.process()


# 2. SEPARACIÓN TRIPARTITA (Train 70% / Val 15% / Test 15%)
# ==========================================
# Sacamos el 15% para el Test final (aislado totalmente)
test_data = 0.15
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=test_data, random_state=42, stratify=y
)
# Del 85% restante, sacamos la validación (0.15 / 0.85)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=test_data/0.85, random_state=42, stratify=y_temp
)

print(f"-> DISTRIBUCIÓN DE DATOS")
print(f"Total de muestras: {len(y)}")
print(f"Train: {len(y_train)} muestras")
print(f"Validation: {len(y_val)} muestras")
print(f"Test: {len(y_test)} muestras\n")


# PARTE A: EL MODELO BASE (Provocando Alta Varianza / Overfitting)
# ==========================================
print("-> MODELO BASE (Sin Regularizar)")
# Modelo muy extenso y especifico forzado dado n_estimators y n_estimators grandes
rf_base = SpotifyRandomForest(
    n_estimators=100, 
    max_depth=40,            
    min_samples_split=2,    
    random_state=42
)
rf_base.fit(X_train, y_train, feature_names=processor.feature_names)

# Evaluamos
y_pred_train_base = rf_base.predict(X_train)
y_pred_val_base = rf_base.predict(X_val)

acc_train_base = accuracy_score(y_train, y_pred_train_base)
acc_val_base = accuracy_score(y_val, y_pred_val_base)

print(f"Accuracy en Train: {acc_train_base:.2%}")
print(f"Accuracy en Validation: {acc_val_base:.2%}")

# Diagnóstico automático 
if acc_train_base > 0.95 and (acc_train_base - acc_val_base) > 0.05:
    print(">> DIAGNÓSTICO: ALTA VARIANZA (OVERFITTING). El modelo memoriza el Train pero pierde precisión en Validación.")
elif acc_train_base < 0.70:
    print(">> DIAGNÓSTICO: ALTO SESGO (UNDERFITTING). El modelo no logra aprender los patrones.")
else:
    print(">> DIAGNÓSTICO: AJUSTE ACEPTABLE (FIT).")

print("\nReporte en Validación (Modelo Base):")
print(classification_report(y_val, y_pred_val_base))



# PARTE B: EL MODELO REGULARIZADO 
# ==========================================
print("\n-> MODELO REGULARIZADO (Técnicas de Ajuste)")
print("Aplicando regularización estricta: bajando max_depth y subiendo min_samples_split")

# Usamos tu misma clase, pero aplicamos regularización con sus hiperparámetros
rf_reg = SpotifyRandomForest(
    n_estimators=200,        # Más árboles para mayor robustez
    max_depth=8,             # REGULARIZACIÓN 1: Poda de profundidad
    min_samples_split=40,    # REGULARIZACIÓN 2: Exige más datos para ramificar
    random_state=42
)
# Nota: la clase ya incluye class_weight="balanced" 
rf_reg.fit(X_train, y_train, feature_names=processor.feature_names)

y_pred_train_reg = rf_reg.predict(X_train)
y_pred_val_reg = rf_reg.predict(X_val)

acc_train_reg = accuracy_score(y_train, y_pred_train_reg)
acc_val_reg = accuracy_score(y_val, y_pred_val_reg)

print(f"Accuracy Regularizado en Train: {accuracy_score(y_train, y_pred_train_reg):.2%}")
print(f"Accuracy Regularizado en Validation: {accuracy_score(y_val, y_pred_val_reg):.2%}")

if acc_train_reg > 0.95 and (acc_train_reg - acc_val_reg) > 0.05:
    print(">> DIAGNÓSTICO: ALTA VARIANZA (OVERFITTING). El modelo memoriza el Train pero pierde precisión en Validación.")
elif acc_train_reg < 0.70:
    print(">> DIAGNÓSTICO: ALTO SESGO (UNDERFITTING). El modelo no logra aprender los patrones.")
else:
    print(">> DIAGNÓSTICO: AJUSTE ACEPTABLE (FIT).")

print("\nReporte en Validación (Modelo Base):")
print(classification_report(y_val, y_pred_val_base))

print("\nReporte en Validación (Modelo Regularizado):")
print(classification_report(y_val, y_pred_val_reg))


# PARTE C: EVALUACIÓN FINAL EN TEST SET
# ==========================================
print("\n-> RENDIMIENTO FINAL EN TEST SET (Datos nunca vistos)")
y_pred_test = rf_reg.predict(X_test)
print(classification_report(y_test, y_pred_test))

+print("\nVisualización exportada")
rf_reg.export_single_tree(tree_index=0, max_depth_viz=8, output_path="docs/rf_tree_viz.png")
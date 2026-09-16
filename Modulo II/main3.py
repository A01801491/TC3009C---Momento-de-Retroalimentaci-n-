import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Importamos mis modulos personalizados
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
if acc_train_base > 0.90 and (acc_train_base - acc_val_base) > 0.05:
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

# Seleccionado de las main3.ipynb (se realizado randomizedserachcv y después grisearchcv)
rf_reg = SpotifyRandomForest(
    criterion='entropy',
    max_depth=4,
    max_features=None,
    min_samples_split=7,
    n_estimators=75,
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

if acc_train_reg > 0.90 and (acc_train_reg - acc_val_reg) > 0.05:
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
print("\n-> RENDIMIENTO FINAL EN TEST SET")
y_pred_test = rf_reg.predict(X_test)
print(classification_report(y_test, y_pred_test))

# PARTE D: COMPARACIÓN DE MATRICES DE CONFUSIÓN (BASE VS REGULARIZADO)
# ==========================================
print("\n-> PARTE D: COMPARACIÓN DE MATRICES DE CONFUSIÓN EN TEST SET")

y_pred_test_base = rf_base.predict(X_test)

# Generamos las matrices de confusión
cm_base = confusion_matrix(y_test, y_pred_test_base)
cm_reg = confusion_matrix(y_test, y_pred_test)  # y_pred_test ya es del rf_reg (Parte C)

# Extracción de métricas - Modelo Base
tn_base, fp_base, fn_base, tp_base = cm_base.ravel()
# Extracción de métricas - Modelo Regularizado
tn_reg, fp_reg, fn_reg, tp_reg = cm_reg.ravel()

print("\n[Modelo Base]")
print(f"Matriz:\n{cm_base}")
print(f" Saltos ignorados (FN): {fn_base}")
print(f" Saltos detectados (TP): {tp_base}")

print("\n[Modelo Regularizado - Optimizado]")
print(f"Matriz:\n{cm_reg}")
print(f" Saltos ignorados (FN): {fn_reg}")
print(f" Saltos detectados (TP): {tp_reg}")

print("\n>> ANÁLISIS DE LA REGULARIZACIÓN:")
diferencia_tp = tp_reg - tp_base
diferencia_fn = fn_base - fn_reg 

if diferencia_tp > 0:
    print(f"- El modelo regularizado logró detectar {diferencia_tp} saltos adicionales (clase 1).")
    print(f"- Redujo los errores de omisión (FN) en {diferencia_fn} casos.")
else:
    print(f"- El modelo regularizado detecto {diferencia_tp} menos saltos que el modelo base (clase 1).")
    print(f"- No redujo los errores de omisión (FN) en {diferencia_fn} casos.")
    print("La regularización estabilizó la varianza y detuvo el overfitting")

print("\nVisualización exportada")
rf_reg.export_single_tree(tree_index=0, max_depth_viz=8, output_path="docs/rf_tree_viz.png")
# main2.py

import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.data_processing import DataProcessor
from src.random_forest import SpotifyRandomForest


def print_confusion_matrix(cm: np.ndarray) -> None:
    """Imprime la matriz de confusión con formato legible en consola."""
    print("\n" + "=" * 44)
    print("         MATRIZ DE CONFUSIÓN")
    print("=" * 44)
    print(f"{'':20} {'Pred 0':>8} {'Pred 1':>8}")
    print(f"{'Real 0 (no skip)':20} {cm[0][0]:>8} {cm[0][1]:>8}")
    print(f"{'Real 1 (skip)':20} {cm[1][0]:>8} {cm[1][1]:>8}")
    print("=" * 44)


def print_feature_importances(rf: SpotifyRandomForest, top_n: int = 10) -> None:
    """Imprime las top_n features más importantes del ensemble."""
    print(f"\n{'='*44}")
    print(f"  TOP {top_n} FEATURES POR IMPORTANCIA (Gini)")
    print(f"{'='*44}")
    for i, (name, score) in enumerate(rf.feature_importances()[:top_n], 1):
        bar = "█" * int(score * 200)
        print(f"  {i:>2}. {name:<35} {score:.4f} {bar}")
    print("=" * 44)


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # 1. Carga y preprocesamiento (mismo pipeline que main.py)
    # ------------------------------------------------------------------
    json_files = glob.glob("data/Streaming_History_Audio_2026*.json")
    processor  = DataProcessor(json_paths=json_files)
    X, y       = processor.process()

    print(f"Shape de X       : {X.shape}")
    print(f"Balance de clases: {y.mean():.2%} positivos (skipped=1)")
    print(f"Clase 0: {(y==0).sum()} muestras | Clase 1: {(y==1).sum()} muestras")

    # ------------------------------------------------------------------
    # 2. Split estratificado (idéntico al de main.py para comparación justa)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras")

    # ------------------------------------------------------------------
    # 3. Entrenamiento del Random Forest con class_weight='balanced'
    # ------------------------------------------------------------------
    print("\nEntrenando Random Forest (n_estimators=200, class_weight='balanced')...")

    rf = SpotifyRandomForest(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
    )
    rf.fit(X_train, y_train, feature_names=processor.feature_names)
    print("¡Entrenamiento completado!")

    # ------------------------------------------------------------------
    # 4. Predicciones y métricas
    # ------------------------------------------------------------------
    y_pred = rf.predict(X_test)

    print_confusion_matrix(confusion_matrix(y_test, y_pred))

    print(f"\n  Accuracy: {accuracy_score(y_test, y_pred):.4f} ({accuracy_score(y_test, y_pred):.2%})")

    print("\n" + "=" * 44)
    print("       REPORTE DE CLASIFICACIÓN")
    print("=" * 44)
    print(classification_report(
        y_test, y_pred,
        target_names=["Clase 0 (no skip)", "Clase 1 (skip)"]
    ))

    # ------------------------------------------------------------------
    # 5. Interpretabilidad del ensemble
    # ------------------------------------------------------------------
    print_feature_importances(rf, top_n=10)
from sklearn.ensemble import RandomForestClassifier
import numpy as np


class SpotifyRandomForest:
    """
    Wrapper sobre RandomForestClassifier de sklearn configurado
    para contrarrestar el desbalance de clases del historial de Spotify.

    El parámetro class_weight='balanced' ajusta internamente el peso
    de cada clase de forma inversamente proporcional a su frecuencia:
        w_j = n_samples / (n_classes * n_samples_j)
    Esto penaliza más los errores sobre la clase minoritaria (skipped=1)
    durante la construcción de cada árbol del ensemble.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 10,
        min_samples_split: int = 5,
        max_features: str = "sqrt",
        random_state: int = 42,
    ):
        """
        Args:
            n_estimators: Número de árboles en el bosque.
            max_depth: Profundidad máxima de cada árbol.
            min_samples_split: Mínimo de muestras para intentar un split.
            max_features: Features evaluadas por split ('sqrt' es estándar CART).
            random_state: Semilla para reproducibilidad.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            max_features=max_features,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,  # usa todos los núcleos disponibles
        )
        self.feature_names: list = []

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list = None):
        """
        Entrena el modelo sobre el conjunto de entrenamiento.

        Args:
            X: Matriz de features (n_samples, n_features).
            y: Vector objetivo binario (0 = no skip, 1 = skip).
            feature_names: Nombres de columnas para interpretabilidad.
        """
        if feature_names is not None:
            self.feature_names = feature_names
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Genera predicciones de clase sobre X.

        Args:
            X: Matriz de features del conjunto de prueba.

        Returns:
            Array de predicciones binarias (0 o 1).
        """
        return self.model.predict(X)

    def feature_importances(self) -> list[tuple]:
        """
        Devuelve las features ordenadas por importancia descendente
        (basada en reducción media de impureza Gini del ensemble).

        Returns:
            Lista de tuplas (nombre_feature, importancia).
        """
        importances = self.model.feature_importances_
        names = self.feature_names if self.feature_names else [
            f"feature_{i}" for i in range(len(importances))
        ]
        ranked = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
        return ranked
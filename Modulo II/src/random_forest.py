from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import tree as sk_tree


class SpotifyRandomForest:
    """
    El parámetro class_weight='balanced' ajusta internamente el peso
    de cada clase de forma inversamente proporcional a su frecuencia:
        w_j = n_samples / (n_classes * n_samples_j)
    Esto penaliza más los errores sobre la clase minoritaria (skipped=1)
    durante la construcción de cada árbol del ensemble.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 15,
        min_samples_split: int = 8,
        max_features: str = "sqrt",
        random_state: int = 42,
    ):
        """
            n_estimators: Número de árboles en el bosque
            max_depth: Profundidad máxima de cada árbol
            min_samples_split: Mínimo de muestras para intentar un split
            max_features: Features evaluadas por split ('sqrt' es estándar CART)
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
        Entrena el modelo sobre el conjunto de entrenamiento
            X: Matriz de features (n_samples, n_features)
            y: Vector objetivo binario (0 = no skip, 1 = skip)
            feature_names: Nombres de columnas para interpretabilidad
        """
        if feature_names is not None:
            self.feature_names = feature_names
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Genera predicciones de clase sobre X

        Args:
            X: Matriz de features del conjunto de prueba

        Returns:
            Array de predicciones binarias (0 o 1)
        """
        return self.model.predict(X)

    def feature_importances(self) -> list[tuple]:
        """
        Devuelve las features ordenadas por importancia descendente
        (basada en reducción media de impureza Gini del ensemble).

        Returns:
            Lista de tuplas (nombre_feature, importancia)
        """
        importances = self.model.feature_importances_
        names = self.feature_names if self.feature_names else [
            f"feature_{i}" for i in range(len(importances))
        ]
        ranked = sorted(zip(names, importances), key=lambda x: x[1], reverse=True)
        return ranked

    def export_single_tree(
        self,
        tree_index: int = 0,
        feature_names: list = None,
        class_names: list = None,
        max_depth_viz: int = 3,
        output_path: str = "docs/arbol_viz.png",
        figsize: tuple = (24, 10),
        dpi: int = 200,
    ) -> None:
        """
        Exporta un árbol individual del ensemble como imagen PNG usando matplotlib.
            tree_index: Índice del árbol a visualizar dentro del ensemble
            feature_names: Nombres de las features para las etiquetas de los nodos
            class_names: Nombres de las clases objetivo
            max_depth_viz: Profundidad máxima a renderizar (3-4 es legible)
            output_path: Ruta de salida incluyendo extensión (.png)
            figsize: Tamaño del canvas en pulgadas
            dpi: Resolución de la imagen exportada.
        """
        from sklearn import tree as sk_tree
        import matplotlib.pyplot as plt

        if not hasattr(self.model, "estimators_"):
            raise RuntimeError(
                "El modelo no ha sido entrenado. Llama a .fit() primero."
            )

        if tree_index >= len(self.model.estimators_):
            raise IndexError(
                f"tree_index={tree_index} fuera de rango. "
                f"El bosque tiene {len(self.model.estimators_)} árboles."
            )

        single_tree  = self.model.estimators_[tree_index]
        names        = feature_names or self.feature_names or None
        target_names = class_names or ["no skip (0)", "skip (1)"]

        fig, ax = plt.subplots(figsize=figsize)

        sk_tree.plot_tree(
            single_tree,
            ax            = ax,
            feature_names = names,
            class_names   = target_names,
            max_depth     = max_depth_viz,
            filled        = True,   # colorea por clase dominante
            rounded       = True,   # bordes redondeados
            impurity      = True,   # muestra Gini en cada nodo
            proportion    = False,  # conteos absolutos, no porcentajes
            precision     = 3,
            fontsize      = 8,
        )

        ax.set_title(
            f"Árbol #{tree_index} del Random Forest  |  "
            f"Profundidad visualizada: {max_depth_viz}",
            fontsize=13,
            fontweight="bold",
            pad=16,
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Árbol #{tree_index} exportado → {output_path}")

        plt.show()
        plt.close(fig)
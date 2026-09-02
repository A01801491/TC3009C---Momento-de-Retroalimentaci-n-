import numpy as np

class Node:
    def __init__(self, feature_index=None,threshold=None, 
                 left=None,right=None,*,value=None):
        """
            feature index - indice de la columna evaluada en el nodo,
            threshold - valor de corte para comparar,
            lef,right - sub nodos
            value - si value != None => el nodo contiene la clase predicha
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        """
            IF value != None devuelve True => el nodo es una hoja
        """
        return self.value is not None

class DecisionTree:
    """
        arbol tipo CART para clasificación binaria (cart:=class and reg tree)
    """
    def __init__(self,max_depth:int=10,min_samples_split:int=2,
                 criterion:str='gini'):
        """
            max_depth - hiperparametro para regularizar
            min_samples_split - minimo de muestras por spllit
            criterion - 'gini'/'entropy' para medir impureza
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def _gini_impurity(self, y: np.ndarray) -> float:
        m = len(y)
        if m == 0:
            return 0.0
        
        counts = np.bincount(y)
        probabilities = counts / m
        
        return 1.0 - np.sum(probabilities ** 2)
    
    def _entropy(self, y: np.ndarray) -> float:
        """Calcula la entropía de Shannon de un conjunto de etiquetas y"""
        m = len(y)
        if m == 0:
            return 0.0
            
        counts = np.bincount(y)
        probabilities = counts / m
        
        probabilities = probabilities[probabilities > 0]
        
        return -np.sum(probabilities * np.log2(probabilities))

    def _impurity(self, y: np.ndarray) -> float:
        if self.criterion == "entropy":
            return self._entropy(y)
        return self._gini_impurity(y)

    def _information_gain(self, y_parent: np.ndarray,
                           y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        impureza(padre) - promedio ponderado de impureza(hijos)
        """
        weight_left = len(y_left) / len(y_parent)
        weight_right = len(y_right) / len(y_parent)
        
        gain = self._impurity(y_parent) - (weight_left * self._impurity(y_left) + weight_right * self._impurity(y_right))
        
        return gain

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Busca, entre todas las columnas y todos los umbrales candidatos,
        el split que maximiza la ganancia de información.
        Devuelve (best_feature_index, best_threshold, best_gain).
        """
        n_samples, n_features = X.shape
        best_gain = -1.0
        best_feature_index = None
        best_threshold = None

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])

            for threshold in thresholds:
                _, y_left, _, y_right = self._split_dataset(
                    X, y, feature_index, threshold
                )

                # Ignora splits degenerados (todo cae de un solo lado)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gain = self._information_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold, best_gain
        

    def _split_dataset(self, X: np.ndarray, y: np.ndarray,
                        feature_index: int, threshold: float) -> tuple:
        
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        return X_left, y_left, X_right, y_right

    def _majority_class(self, y: np.ndarray) -> int:
        """Devuelve la clase mayoritaria (0 o 1) dentro de y."""
        counts = np.bincount(y.astype(int))
        return int(np.argmax(counts))    


    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Construye el árbol recursivamente.
        a. Casos base - profundidad máxima alcanzada, nodo puro,
        o muestras insuficientes -> crea un Node hoja.
        b. Caso recursivo - encuentra el mejor split y construye
        subárboles izquierdo/derecho.
        """

        n_samples = len(y)

        # --- Condiciones de parada ---
        if (depth >= self.max_depth
                or n_samples < self.min_samples_split
                or self._impurity(y) == 0):
            leaf_value = self._majority_class(y)
            return Node(value=leaf_value)

        feature_index, threshold, gain = self._best_split(X, y)

        # Si ningún split mejora la impureza, se detiene la recursión
        if feature_index is None or gain <= 0:
            leaf_value = self._majority_class(y)
            return Node(value=leaf_value)

        X_left, y_left, X_right, y_right = self._split_dataset(
            X, y, feature_index, threshold
        )
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        return Node(
            feature_index=feature_index,
            threshold=threshold,
            left=left_subtree,
            right=right_subtree
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self._build_tree(X, y, depth=0)
        return self
    def _predict_single(self, x: np.ndarray, node: Node):        
        if node.is_leaf():
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self._predict_single(x, self.root) for x in X])
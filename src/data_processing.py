import pandas as pd
import numpy as np

class DataProcessor:
    """
    Encargada de transformar el historial crudo de Spotify (JSON -> DataFrame)
    en matrices X, y listas para alimentar el DecisionTree hecho desde cero.
    """

    ONE_HOT_COLUMNS = ["platform", "reason_start", "conn_country"]
    ARTIST_COLUMN = "master_metadata_album_artist_name"
    TARGET_COLUMN = "skipped"
    TIMESTAMP_COLUMN = "ts"
    TIMEZONE = "America/Mexico_City"

    def __init__(self, json_paths: list):
        self.json_paths = json_paths
        self.df = None

    def load_json(self):
        """Carga y concatena todos los archivos JSON en un solo DataFrame."""
        frames = [pd.read_json(path) for path in self.json_paths]
        self.df = pd.concat(frames, ignore_index=True)
        return self.df

    def _convert_timestamp(self):
        """Convierte 'ts' a America/Mexico_City y extrae hour y day_of_week."""
        self.df[self.TIMESTAMP_COLUMN] = pd.to_datetime(
            self.df[self.TIMESTAMP_COLUMN], utc=True
        )
        local_ts = self.df[self.TIMESTAMP_COLUMN].dt.tz_convert(self.TIMEZONE)

        self.df["hour"] = local_ts.dt.hour
        self.df["day_of_week"] = local_ts.dt.dayofweek

    def _clean_target(self):
        """Asegura que 'skipped' sea 0/1 y elimina registros con target nulo."""
        self.df = self.df[self.df[self.TARGET_COLUMN].notna()]
        self.df[self.TARGET_COLUMN] = self.df[self.TARGET_COLUMN].astype(int)

    def _frequency_encode_artist(self):
        """
        Reemplaza el nombre del artista por su conteo total de
        reproducciones en el año (Frequency Encoding).
        """
        freq_map = self.df[self.ARTIST_COLUMN].value_counts()
        self.df[self.ARTIST_COLUMN] = self.df[self.ARTIST_COLUMN].map(freq_map)
        self.df[self.ARTIST_COLUMN] = self.df[self.ARTIST_COLUMN].fillna(0)

    def _one_hot_encode(self):
        """Aplica One-Hot Encoding a platform, reason_start y conn_country."""
        self.df = pd.get_dummies(
            self.df, columns=self.ONE_HOT_COLUMNS,
            prefix=self.ONE_HOT_COLUMNS, dtype=int
        )

    def _select_final_features(self):
        """
        LISTA BLANCA: Extrae estrictamente las variables autorizadas.
        Ignora automáticamente basura como audiolibros, ip_addr o shuffle.
        """
        one_hot_cols = [c for c in self.df.columns if any(c.startswith(p + "_") for p in self.ONE_HOT_COLUMNS)]
        
        # Filtramos solo lo que nos sirve matemáticamente
        features = [self.TARGET_COLUMN, self.ARTIST_COLUMN, "hour", "day_of_week"] + one_hot_cols
        self.df = self.df[features]

    def process(self):
        """
        Orquesta el pipeline completo y devuelve (X, y) como numpy arrays.
        """
        if self.df is None:
            self.load_json()

        self._convert_timestamp()
        self._clean_target()
        self._frequency_encode_artist()
        self._one_hot_encode()
        
        # Aplicamos la lista blanca ANTES del dropna
        self._select_final_features()
        
        # Limpieza final segura
        self.df = self.df.dropna()

        # Separación en X e y asegurando formato numérico
        y = self.df[self.TARGET_COLUMN].to_numpy(dtype=int)
        X = self.df.drop(columns=[self.TARGET_COLUMN]).to_numpy(dtype=float)

        self.feature_names = self.df.drop(
            columns=[self.TARGET_COLUMN]
        ).columns.tolist()

        return X, y
# Posicionamiento de Precio en Línea Blanca — Marca KRONOS

Modelo estadístico base (regresión lineal múltiple, OLS) para decidir el posicionamiento de precio de KRONOS frente a la competencia, evaluando el impacto conjunto de la distribución ponderada, las promociones, los quiebres de inventario y la estacionalidad sobre el **Market Share**.

## Objetivo

Responder, con base en datos históricos, cómo cada palanca comercial (precio relativo, cobertura de distribución, promociones, desabasto y temporada) afecta la participación de mercado de KRONOS, y estimar el Market Share esperado bajo un escenario de negocio específico, con su respectivo intervalo de confianza e intervalo de predicción al 95%.

## Estructura del repositorio

```
.
├── data/
│   └── datos_posicionamiento_kronos.csv   # Dataset histórico (ventas, precios, distribución, etc.)
├── Portafolio_Construcción_de_un_modelo_estadístico_base.ipynb
└── README.md
```

- **`data/`**: contiene el archivo `datos_posicionamiento_kronos.csv` con las observaciones históricas usadas para entrenar y evaluar el modelo. Si el archivo no se encuentra en esta ruta, el notebook genera automáticamente un DataFrame de ejemplo (fallback) para que el código siga siendo ejecutable.
- **`Portafolio_Construcción_de_un_modelo_estadístico_base.ipynb`**: notebook principal con todo el análisis, de EDA a conclusiones de negocio.

## Variables del modelo

| Tipo | Variable | Descripción |
|---|---|---|
| Dependiente (Y) | `Market_Share` | Participación de mercado de KRONOS (%) |
| Independiente (X) | `Índice_Precio` | Precio KRONOS / Precio Competencia |
| Independiente (X) | `Dist_Ponderada` | Cobertura de distribución ponderada |
| Independiente (X) | `Promoción` | 1 si hubo promoción, 0 si no |
| Independiente (X) | `Quiebre` | Proporción de desabasto de inventario |
| Independiente (X) | `Es_Temporada_Alta` | 1 si el mes es noviembre o diciembre, 0 en otro caso (derivada de `Fecha`) |
| Independiente (X) | `Región_*` | Dummies de región (Norte, Occidente, Sur); *Centro* es la categoría de referencia |

Variables como `Precio_KRONOS`, `Precio_Competencia`, `Dist_Numérica`, `Ventas_KRONOS`, `Unidades_KRONOS` y `Contribución` se excluyen deliberadamente del modelo por redundancia (colinealidad con `Índice_Precio` / `Dist_Ponderada`) o por fuga de datos (son consecuencia, no causa, del Market Share).

## Contenido del notebook

0. **EDA** — distribución univariada y bivariada de las variables, matriz de correlación.
1. **Ingeniería de características y dummies** — extracción de `Es_Temporada_Alta`, cálculo de `Índice_Precio`, dummies de `Región` (evitando la trampa de la variable dummy).
2. **Transformación y escalamiento** — modelo OLS con variables originales (coeficientes reales) y modelo OLS con variables estandarizadas vía `StandardScaler` (coeficientes Beta, importancia relativa).
3. **Evaluación econométrica** — $R^2$, $R^2$ ajustado, p-values, VIF (multicolinealidad) y validación de supuestos (normalidad y autocorrelación de residuos).

** **Predicción de escenario** — Market Share esperado para Región Centro, Temporada Alta, `Índice_Precio = 0.98`, `Dist_Ponderada = 0.80`, `Promoción = 1`, con su Intervalo de Confianza (95%) e Intervalo de Predicción (95%).

** **Análisis de dummies** — regresión simple con la variable de mayor impacto (`Índice_Precio`) + dummies de región, para aislar el efecto geográfico del efecto de precio.

## Requisitos

```
pandas
numpy
statsmodels
scikit-learn
matplotlib
seaborn
```

Instalación rápida:

```bash
pip install pandas numpy statsmodels scikit-learn matplotlib seaborn
```

## Cómo ejecutar

1. Clonar el repositorio.
2. Verificar que `data/datos_posicionamiento_kronos.csv` exista en la ruta indicada.
3. Abrir `Portafolio_Construcción_de_un_modelo_estadístico_base.ipynb` en Jupyter y ejecutar las celdas en orden.

## Autor

David Tinoco Romero — A01801491
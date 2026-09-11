# Árbol de Decisión CART desde Cero: Predicción de Saltos en Spotify 

**Autor:** David Tinoco Romero  
**Institución:** Tecnológico de Monterrey (Campus Estado de México)  
**Curso:** Inteligencia Artificial Avanzada (Módulo II)  
**Entregable:** Implementación de una técnica de aprendizaje máquina sin el uso de un framework.

## Descripción del Proyecto

Este repositorio contiene la implementación nativa de un algoritmo de **Árbol de Decisión (CART)** desarrollado completamente desde cero utilizando Programación Orientada a Objetos en Python y álgebra matricial con `NumPy`. 

El objetivo del modelo es clasificar de forma binaria si una canción será saltada (`skipped = 1`) o escuchada completamente (`skipped = 0`), utilizando un conjunto de datos empírico de mi propio historial de reproducciones de Spotify (2026). 

**Restricción Técnica Principal:** 
En estricto apego a la rúbrica, **no se utilizó ningún framework de Machine Learning (como scikit-learn)** para la lógica de partición, cálculo de impureza (Gini/Entropía), ganancia de información, entrenamiento (`fit`) o inferencia (`predict`). Las librerías de alto nivel se reservaron única y exclusivamente para la ingesta de datos, separación de subconjuntos y evaluación de métricas.

## Arquitectura del Repositorio

```text
├── data/                  # Archivos JSON crudos (Streaming History 2026)
├── docs/                  # Documentación adicional y notas
├── notebooks/             # Entornos interactivos secundarios de exploración (EDA)
├── src/                   # Código fuente modular
│   ├── data_processing.py # Pipeline de limpieza, ingeniería de características (Frecuency & One-Hot Encoding) y Lista Blanca
│   └── decision_tree.py   # Lógica matemática central del Árbol y la clase Node
├── .gitignore             # Archivos excluidos del control de versiones
├── LICENSE                # Licencia del repositorio
├── Momento de Retroalimentación_ Módulo 2... .pdf  # Reporte formal y análisis de resultados
├── README.md              # Documentación del proyecto (Este archivo)
├── main.ipynb             # Notebook para la visualización de métricas (Matriz de Confusión) y estructura gráfica del árbol
└── main.py                # Script principal de ejecución nativa por consola

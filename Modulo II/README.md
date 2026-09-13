# Predicción de Saltos en Spotify: Árbol de Decisión (CART) Nativo y Random Forest 

**Autor:** David Tinoco Romero  
**Institución:** Tecnológico de Monterrey (Campus Estado de México)  
**Curso:** Inteligencia Artificial Avanzada (Módulo II)  
**Entregable:** Portafolio de Implementación de Modelos de Aprendizaje Máquina (Con y sin Frameworks).

## 📌 Descripción del Proyecto

Este repositorio documenta la evolución de un pipeline de Machine Learning cuyo objetivo es clasificar de forma binaria si una canción será saltada (`skipped = 1`) o escuchada completamente (`skipped = 0`), basándose en un conjunto de datos empírico de mi propio historial de reproducciones de Spotify (2026).

El proyecto se divide en diferentes fases de entrega, escalando desde la programación pura de algoritmos hasta el uso de modelos ensamblados comerciales:

### **Fase 1 (Entrega 1) - Árbol de Decisión CART Nativo:** 

Implementación de un algoritmo de clasificación desarrollado completamente desde cero utilizando Programación Orientada a Objetos en Python y álgebra matricial con `NumPy`. **Restricción Técnica:** En estricto apego a la rúbrica inicial, *no se utilizó ningún framework* para la lógica de partición, cálculo de impureza o entrenamiento.

#### Intrucciones de uso: Resultados de CART
Para ejecutar el primer modelo sin el uso de framework:
```bash
python main.py
```

### **Fase 2 (Entrega 2) - Random Forest con Framework:** 

Implementación de un modelo ensamblado (Random Forest) empleando librerías de alto nivel (como `scikit-learn`). El objetivo de esta fase es contrastar la capacidad predictiva, la mitigación del sobreajuste (overfitting) y el manejo del desbalance de clases frente al modelo manual desarrollado en la primera etapa.

#### Intrucciones de uso: Resiltados Random Forest
Para ejecutar el segundo modelo con el uso de framework:
```bash
python main2.py
```

### **Fase 3 (Entrega 3) - Diagnóstico de Desempeño y Regularización:** 

Análisis profundo de la varianza, el sesgo (bias) y el nivel de ajuste (fitting) del modelo Random Forest. Esta etapa implementa una partición tripartita (Entrenamiento, Validación y Prueba) para diagnosticar el comportamiento del algoritmo frente a datos invisibles. Posteriormente, se aplican técnicas de regularización (control de profundidad, poda de hojas y balanceo de pesos de clases) para mitigar el sobreajuste y mejorar la sensibilidad hacia la clase minoritaria (canciones saltadas).

#### Intrucciones de uso: Diagnóstico y Regularización (Train/Val/Test)
Para ejecutar el diagnóstico de varianza/sesgo y entrenar el modelo regularizado:
```bash
python main3.py
```

## 📂 Arquitectura del Repositorio

```bash
├── data/                  # Archivos JSON crudos 
│   └── Streaming_History_Audio... # Datos ocultos por privacidad
├── docs/                  # Reportes de entrega PDF y elementos de los anexos
│   ├── Resultados del ... # PDF de la primera entrega (CART Manual)
│   ├── Resultados de ...2 # PDF de la segunda entrega (Random Forest)
│   └── Resultados de ...3 # PDF de la tercera entrega
├── notebooks/             # Entornos interactivos secundarios de exploración (EDA)
├── src/                   # Código fuente modular
│   ├── data_processing.py # Pipeline de limpieza, ingeniería de características (Frecuency & One-Hot Encoding) y Lista Blanca
│   ├── decision_tree.py   # Lógica matemática central del Árbol manual y la clase Node
│   └── random_forest.py   # Lógica y configuración del modelo ensamblado usando framework
├── .gitignore             # Archivos excluidos del control de versiones
├── LICENSE                # Licencia del repositorio
├── README.md              # Documentación del proyecto (Este archivo)
├── main.py                # Script principal de ejecución (Entrega 1 - CART)
├── main2.py               # Script principal de ejecución (Entrega 2 - Random Forest)
└── main3.py               # Script principal de ejecución (Entrega 3)
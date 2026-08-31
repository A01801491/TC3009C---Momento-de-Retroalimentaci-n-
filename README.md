# TC3009C---Momento-de-Retroalimentaci-n-
Repositorio para la entrega de Momento de Retroalimentación para el módulo II de la materia TC3009C

## Arquitectura 
TC3009C---Momento-de-Retroalimentaci-ny/
│
├── data/                  # Aquí ya vive tu JSON. Todo lo de aquí es ignorado por Git.
│   └── Streaming_History_Audio_2026.json
│
├── src/                   # (Source) Aquí vivirá toda la lógica de tu código.
│   ├── __init__.py        # Archivo vacío para que Python reconozca la carpeta.
│   ├── data_processing.py # Funciones para leer el JSON y convertir texto a números.
│   ├── metrics.py         # Tus cálculos matemáticos (Matriz de confusión, Accuracy, Gini).
│   └── decision_tree.py   # La clase con la programación orientada a objetos de tu algoritmo.
│
├── docs/                  # Carpeta para la documentación.
│   └── reporte_final.pdf  # El PDF con los resultados y métricas que te pide la rúbrica.
│
├── .gitignore             # El archivo de protección que ya configuramos.
├── README.md              # La portada de tu proyecto.
└── main.py                # El script principal que unirá todo y correrá en consola.
# Estructura de Outputs - Análisis Exploratorio

Este documento describe la organización de todos los archivos generados por el análisis exploratorio.

## Estructura de Carpetas

```
outputs/
├── datos/
│   └── dataset_victimas_limpio.csv          # Dataset procesado y limpio
│
└── graficos/
    ├── analisis/
    │   ├── normalidad.png                    # Análisis de normalidad de variables
    │   └── exploratorios.png                 # Gráficos exploratorios generales
    │
    ├── preguntas/
    │   ├── pregunta1_patrones_temporales.png # Pregunta 1: Patrones temporales
    │   ├── pregunta2_delitos_sexo.png        # Pregunta 2: Delitos por sexo
    │   ├── pregunta3_concentracion_geografica.png # Pregunta 3: Concentración geográfica
    │   ├── pregunta4_estacionalidad_mensual.png   # Pregunta 4: Estacionalidad mensual
    │   └── pregunta5_delitos_sexuales.png    # Pregunta 5: Delitos sexuales
    │
    └── clustering/
        ├── metodo_codo_dendrograma.png       # Método del codo y dendrograma
        ├── caracterizacion.png               # Caracterización de clusters
        └── clustermap.png                    # Clustermap integrado

```

## Uso

Ejecutar el análisis completo:

```powershell
python analisis_exploratorio.py
```

El script automáticamente:
1. Crea la estructura de carpetas si no existe
2. Procesa y limpia los datos
3. Genera todos los gráficos organizados
4. Guarda el dataset limpio

## Archivos Generados

### datos/
- **dataset_victimas_limpio.csv**: Dataset procesado con variables unificadas y limpias

### graficos/analisis/
- **normalidad.png**: Pruebas de normalidad (Q-Q plots, histogramas)
- **exploratorios.png**: Visualizaciones generales del dataset

### graficos/preguntas/
Contiene las visualizaciones que responden a las 5 preguntas de investigación del proyecto

### graficos/clustering/
- **metodo_codo_dendrograma.png**: Selección de K óptimo + dendrograma jerárquico
- **caracterizacion.png**: Análisis detallado de los clusters (edad, sexo, hora, tamaño)
- **clustermap.png**: Visualización integrada con dendrograma y heatmap

## Notas

- Todos los gráficos se guardan en alta resolución (300 DPI)
- La estructura se crea automáticamente al ejecutar el script
- Los archivos existentes se sobrescriben en cada ejecución
- El formato de salida de todos los gráficos es PNG

## Para empezar

1. Asegúrate de tener el dataset: `dataset_victimas.csv`
2. Instala las dependencias: `pip install -r requirements.txt`
3. Ejecuta el análisis: `python analisis_exploratorio.py`
4. Revisa los outputs en la carpeta `outputs/`

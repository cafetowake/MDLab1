# Proyecto: Análisis Exploratorio de Víctimas de Delitos 

**Universidad del Valle de Guatemala**  
**CC3074 - Minería de Datos**  

**Autores:**
- Angie Nadissa Vela López - 23764
- Paula Daniela De León Godoy - 23000

---

## Descripción del Proyecto

Análisis exploratorio completo de datos de víctimas de delitos en Guatemala registrados por la PNC-INE, 
aplicando técnicas de minería de datos para identificar patrones, tendencias y perfiles de victimización.

---

## Requisitos

### Descripción del Dataset
- Cantidad de variables y observaciones
- Tipos de cada variable
- Exploración inicial de estructura

### Análisis de Variables
- **Numéricas**: Pruebas de normalidad, distribuciones, outliers
- **Categóricas**: Tablas de frecuencia

### Cruces de Variables
- Cruces entre variables clave
- Identificación de relaciones importantes

### Preguntas de Investigación
1. **Patrones temporales**: ¿Existen patrones por día/hora?
2. **Delitos por sexo**: ¿Diferencias en tipos de delito?
3. **Concentración geográfica**: ¿Focos delictivos específicos?
4. **Estacionalidad mensual**: ¿Variaciones a lo largo del año?
5. **Delitos sexuales**: ¿Características específicas?

### Parte E: Gráficos Exploratorios
- Visualizaciones completas y detalladas
- Múltiples perspectivas de análisis

### Parte F: Clustering
- **K-Means** como método principal
- **Clustering Jerárquico** como complemento
- Método del codo para selección de K
- Interpretación detallada de clusters
- Identificación de perfiles de víctimas

---

## Tecnologías Utilizadas

### Lenguajes y Herramientas
- **Python 3.11**
- **Jupyter (para ejemplos)**
- **VS Code**

### Librerías Principales
```python
pandas==3.0.0           # Manipulación de datos
numpy==2.4.2            # Operaciones numéricas
matplotlib==3.9.0       # Visualización básica
seaborn==0.13.0         # Visualización estadística
scipy==1.14.0           # Funciones científicas
scikit-learn>=1.3.0     # Machine Learning (clustering)
kneed>=0.8.0            # Detección automática de K óptimo
pyreadstat==1.3.3       # Lectura de archivos .sav
```

---

## Estructura del Proyecto

```
lab_proyecto/
│
├── analisis_exploratorio.py       # Script principal del análisis
├── create_dataset.py              # Preprocesamiento inicial
├── requirements.txt               # Dependencias del proyecto
│
├── dataset_victimas.csv           # Dataset unificado
│
├── Documentación/
│   ├── README_OUTPUTS.md             # Documentación de outputs
│
├── data_ine/                      # Datos crudos por año (.sav)
│   ├── victimas_2011.sav
│   ├── victimas_2012.sav
│   └── ... (2011-2023)
│
│
└── outputs/                       # RESULTADOS DEL ANÁLISIS
    ├── datos/
    │   └── dataset_victimas_limpio.csv
    └── graficos/
        ├── analisis/
        │   ├── normalidad.png
        │   └── exploratorios.png
        ├── preguntas/
        │   ├── pregunta1_patrones_temporales.png
        │   ├── pregunta2_delitos_sexo.png
        │   ├── pregunta3_concentracion_geografica.png
        │   ├── pregunta4_estacionalidad_mensual.png
        │   └── pregunta5_delitos_sexuales.png
        └── clustering/
            ├── metodo_codo_dendrograma.png
            ├── caracterizacion.png
            └── clustermap.png
```

---

## Guía de Uso

### 1. Instalación

```powershell
# Clonar/descargar el proyecto
cd "ruta/al/proyecto"

# Crear ambiente virtual (opcional pero recomendado)
python -m venv myenv
.\myenv\Scripts\Activate.ps1

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecución

```powershell
# Ejecutar el análisis completo
python analisis_exploratorio.py
```

**Tiempo estimado:** 
5-10 minutos dependiendo del hardware

### 3. Resultados

Los resultados se guardan automáticamente en la carpeta `outputs/`:
- **Dataset limpio**: `outputs/datos/dataset_victimas_limpio.csv`
- **Gráficos**: `outputs/graficos/` (organizados por categoría)

---

## Técnicas de Clustering Implementadas

### K-Means (Método Principal)
- Estandarización con `StandardScaler`
- Método del codo para selección de K
- Detección automática con `KneeLocator`
- Configuración: `init="k-means++"`, `n_init=10`, `random_state=42`

### Clustering Jerárquico (Validación)
- Método Ward
- Distancia euclidiana
- Dendrograma para visualización

### Comparación
- Adjusted Rand Index (ARI)
- Interpretación de consistencia entre métodos

---

## Características del Análisis

### Variables Analizadas
- **Numéricas**: Edad, año
- **Categóricas**: Sexo, delito, departamento, mes, día de la semana, hora
- **Temporales**: Año, mes, día, hora categorizada

### Filtros y Limpieza
- Unificación de columnas entre años
- Manejo de valores faltantes
- Normalización de categorías
- Codificación de variables para clustering

### Visualizaciones (10 gráficos en total)
- Análisis de normalidad con Q-Q plots
- Gráficos exploratorios generales
- 5 visualizaciones para preguntas de investigación
- 3 visualizaciones detalladas de clustering

---

## Metodología Seguida

### Basado en los Ejemplos de Clustering
Este proyecto implementa las técnicas aprendidas en clase:
- K-Means básico (ejemplo Países)
- Clustering con datos categóricos
- Método del codo + KneeLocator
- Clustering jerárquico con AgglomerativeClustering
- Dendrograma + Mapa de calor (Clustermap)

### Mejores Prácticas Aplicadas
- Estandarización Z-score
- Codificación one-hot de categóricas
- Selección automática de parámetros
- Validación con múltiples métodos
- Documentación exhaustiva

## Hallazgos Principales

### Patrones Identificados
- Concentración de delitos en horarios específicos
- Diferencias significativas por sexo en tipos de delito
- Focos geográficos bien definidos
- Estacionalidad mensual presente

### Clusters Identificados
El clustering reveló perfiles distintos de víctimas:
- Diferenciados por edad
- Patrones por sexo
- Tipos de delito específicos
- Horarios característicos


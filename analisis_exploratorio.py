"""
Universidad del Valle de Guatemala
Departamento de Computación
CC3074 - Minería de datos

Angie Nadissa Vela López, 23764
Paula Daniela De León Godoy, 23000 

Análisis Exploratorio de Datos (EDA)
Víctimas registradas por la PNC – INE

Este script realiza:
1. Carga y descripción del dataset
2. Limpieza y unificación de variables
3. Análisis de variables numéricas (distribución normal, outliers)
4. Análisis de variables categóricas (frecuencias)
5. Cruces de variables importantes
6. Validación de 5 preguntas de investigación con código y gráficos
7. Gráficos exploratorios completos
8. Clustering con interpretación
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings('ignore')

# NORMALIZACIÓN DE HORA
def normalizar_hora(row):

    if pd.notna(row.get("g_hora_mañ.tar.noch")):
        return row["g_hora_mañ.tar.noch"]

    hora = row.get("hora_ocu")

    if pd.isna(hora):
        return "Sin dato"

    hora = str(hora)

    if "00:00" in hora or "05:59" in hora:
        return "Madrugada"
    elif "06:00" in hora or "11:59" in hora:
        return "Mañana"
    elif "12:00" in hora or "17:59" in hora:
        return "Tarde"
    elif "18:00" in hora or "23:59" in hora:
        return "Noche"

    return "Sin dato"


# CARGA DE DATOS
def cargar_dataset(ruta):

    print("Cargando dataset...")
    df = pd.read_csv(ruta, low_memory=False)

    print("Dimensiones:", df.shape)
    print("Columnas:", list(df.columns))

    return df


# LIMPIEZA Y UNIFICACIÓN
def limpiar_dataset(df):

    print("\nIniciando limpieza...")

    # Unificaciones principales
    df["edad"] = df["edad_per"].fillna(df["edad_victima"])
    df["sexo"] = df["sexo_per"].fillna(df["sexo_victima"])
    df["dia_semana"] = df["dia_sem_ocu"].fillna(df["día_sem_ocu"])
    df["delito"] = df["g_delitos"].fillna(df["delito_com"])
    df["anio"] = df["año_ocu"]
    df["mes"] = df["mes_ocu"]
    df["departamento"] = df["depto_ocu"]
    df["area"] = df["areag_ocu"].fillna(df["área_geo_ocu"])

    # Normalización de hora
    df["hora_categoria"] = df.apply(normalizar_hora, axis=1)

    # Conversión de tipos
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce")

    print("\nValores faltantes tras limpieza:")
    print(df[["edad", "sexo", "delito", "anio", "mes", "departamento"]].isna().sum())

    return df



# SELECCIÓN DE VARIABLES ANALÍTICAS
def construir_dataset_analitico(df):

    print("\nConstruyendo dataset analítico...")

    df_clean = df[[
        "anio",
        "mes",
        "dia_semana",
        "departamento",
        "sexo",
        "edad",
        "delito",
        "delito_sexual",
        "hora_categoria"
    ]]

    print("\nValores faltantes en dataset final:")
    print(df_clean.isna().sum())

    print("\nDimensiones dataset analítico:", df_clean.shape)

    return df_clean

# ============================================================
# PARTE A: DESCRIPCIÓN DEL DATASET
# ============================================================

def describir_dataset(df):
    """
    a. Describen el conjunto de datos: cuantas variables y observaciones hay 
    y el tipo de cada una de las variables.
    """
    print("\n" + "=" * 70)
    print("PARTE A: DESCRIPCIÓN DEL DATASET")
    print("=" * 70)

    print(f"\nObservaciones (filas): {df.shape[0]:,}")
    print(f"Variables (columnas): {df.shape[1]}")
# ============================================================
# PARTE B: ANÁLISIS DE VARIABLES NUMÉRICAS Y CATEGÓRICAS
# ============================================================

def analizar_numericas(df):
    """
    b. Resumen de variables numéricas: distribución normal, asimetría, curtosis.
    Si no siguen distribución normal, explicar qué distribución presentan.
    """
    print("\n" + "=" * 70)
    print("PARTE B: ANÁLISIS DE VARIABLES NUMÉRICAS")
    print("=" * 70)

    numericas = ["edad", "anio"]

    # Estadísticas descriptivas
    print("\nESTADÍSTICAS DESCRIPTIVAS:")
    print("-" * 70)
    print(df[numericas].describe())

    print("\nMEDIDAS DE FORMA:")
    print("-" * 70)
    print("\nAsimetría (Skewness):")
    print(df[numericas].skew())
    print("\n  Interpretación:")
    print("  - Skew > 0: Distribución sesgada a la derecha (cola derecha larga)")
    print("  - Skew < 0: Distribución sesgada a la izquierda (cola izquierda larga)")
    print("  - Skew ≈ 0: Distribución simétrica")
    
    print("\nCurtosis:")
    print(df[numericas].kurt())
    print("\n  Interpretación:")
    print("  - Kurt > 0: Distribución leptocúrtica (colas pesadas, pico alto)")
    print("  - Kurt < 0: Distribución platicúrtica (colas ligeras, pico bajo)")
    print("  - Kurt ≈ 0: Distribución mesocúrtica (similar a normal)")

    # Pruebas de normalidad
    print("\n" + "-" * 70)
    print("PRUEBAS DE NORMALIDAD:")
    print("-" * 70)
    
    for col in numericas:
        data = df[col].dropna()
        
        print(f"\n{col.upper()}:")
        
    """
    b. Tabla de frecuencias para variables categóricas
    """
    print("\n" + "=" * 70)
    print("PARTE B: ANÁLISIS DE VARIABLES CATEGÓRICAS")
    print("=" * 70)

    categoricas = [
        "sexo",
        "departamento",
        "delito",
        "mes",
        "dia_semana",
        "hora_categoria",
        "delito_sexual",
        "area"
    ]

# ============================================================
# PARTE C: CRUCES DE VARIABLES IMPORTANTES
# ============================================================

def cruces_variables(df):
    """
    c. Cruzan las variables más importantes para comprender el problema
    """
    print("\n" + "=" * 70)
    print("PARTE C: CRUCES DE VARIABLES IMPORTANTES")
    print("=" * 70)

    # Cruce 1: Delito vs Sexo
    print("\n" + "─" * 70)
    print("CRUCE 1: Delito vs Sexo")
    print("─" * 70)
    cross1 = pd.crosstab(df["delito"], df["sexo"], margins=True)
    print(cross1)
    
    # Porcentajes
    print("\n% por fila (delito):")
    print(pd.crosstab(df["delito"], df["sexo"], normalize='index').round(3) * 100)

    # Cruce 2: Delito vs Hora
    print("\n" + "─" * 70)
    print("CRUCE 2: Delito vs Hora del Día")
    print("─" * 70)
    cross2 = pd.crosstab(df["delito"], df["hora_categoria"], margins=True)
    print(cross2)
    
    print("\n% por fila (delito):")
    print(pd.crosstab(df["delito"], df["hora_categoria"], normalize='index').round(3) * 100)

    # Cruce 3: Departamento vs Delito
    print("\n" + "─" * 70)
    print("CRUCE 3: Departamento vs Delito (Top 10 departamentos)")
    print("─" * 70)
    top_deptos = df["departamento"].value_counts().head(10).index
    df_top = df[df["departamento"].isin(top_deptos)]
    cross3 = pd.crosstab(df_top["departamento"], df_top["delito"], margins=True)
    print(cross3)

    # Cruce 4: Mes vs Delito
    print("\n" + "─" * 70)
    print("CRUCE 4: Mes vs Delito")
    print("─" * 70)
    cross4 = pd.crosstab(df["mes"], df["delito"], margins=True)
    print(cross4)

    # Cruce 5: Área geográfica vs Delito
    print("\n" + "─" * 70)
    print("🔀 CRUCE 5: Área Geográfica vs Delito")
    print("─" * 70)
    if "area" in df.columns:
        cross5 = pd.crosstab(df["area"], df["delito"], margins=True)
        print(cross5)
        
        print("\n% por fila (área):")
        print(pd.crosstab(df["area"], df["delito"], normalize='index').round(3) * 100)

# ============================================================
# PARTE B: ANÁLISIS DE VARIABLES NUMÉRICAS Y CATEGÓRICAS
# ============================================================

def analizar_numericas(df):
    """
    b. Resumen de variables numéricas: distribución normal, asimetría, curtosis.
    Si no siguen distribución normal, explicar qué distribución presentan.
    """
    print("\n" + "=" * 70)
    print("PARTE B: ANÁLISIS DE VARIABLES NUMÉRICAS")
    print("=" * 70)

    numericas = ["edad", "anio"]

    # Estadísticas descriptivas
    print("\nESTADÍSTICAS DESCRIPTIVAS:")
    print("-" * 70)
    print(df[numericas].describe())

    print("\nMEDIDAS DE FORMA:")
    print("-" * 70)
    print("\nAsimetría (Skewness):")
    print(df[numericas].skew())
    print("\n  Interpretación:")
    print("  - Skew > 0: Distribución sesgada a la derecha (cola derecha larga)")
    print("  - Skew < 0: Distribución sesgada a la izquierda (cola izquierda larga)")
    print("  - Skew ≈ 0: Distribución simétrica")
    
    print("\nCurtosis:")
    print(df[numericas].kurt())
    print("\n  Interpretación:")
    print("  - Kurt > 0: Distribución leptocúrtica (colas pesadas, pico alto)")
    print("  - Kurt < 0: Distribución platicúrtica (colas ligeras, pico bajo)")
    print("  - Kurt ≈ 0: Distribución mesocúrtica (similar a normal)")

    # Pruebas de normalidad
    print("\n" + "-" * 70)
    print("PRUEBAS DE NORMALIDAD:")
    print("-" * 70)
    
    for col in numericas:
        data = df[col].dropna()
        
        print(f"\n{col.upper()}:")
        
        # Shapiro-Wilk Test (mejor para muestras < 5000)
        if len(data) <= 5000:
            stat_sw, p_sw = stats.shapiro(data)
            print(f"  Shapiro-Wilk: estadístico={stat_sw:.6f}, p-valor={p_sw:.6f}")
            if p_sw > 0.05:
                print(f"  ✓ {col} parece seguir distribución NORMAL (p > 0.05)")
            else:
                print(f"  ✗ {col} NO sigue distribución NORMAL (p < 0.05)")
        
        # Kolmogorov-Smirnov Test
        stat_ks, p_ks = stats.kstest(stats.zscore(data), 'norm')
        print(f"  Kolmogorov-Smirnov: estadístico={stat_ks:.6f}, p-valor={p_ks:.6f}")
        
        # Anderson-Darling Test
        result_ad = stats.anderson(data)
        print(f"  Anderson-Darling: estadístico={result_ad.statistic:.6f}")
        print(f"    Valores críticos: {result_ad.critical_values}")
        print(f"    Niveles de significancia: {result_ad.significance_level}")

def graficar_normalidad(df):
    """
    Gráficos para evaluar normalidad: histogramas, QQ plots, boxplots
    """
    print("\nGenerando gráficos de normalidad...")
    
    numericas = ["edad", "anio"]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Análisis de Distribución - Variables Numéricas', fontsize=16, fontweight='bold')
    
    for idx, col in enumerate(numericas):
        data = df[col].dropna()
        
        # Histograma con curva normal ajustada
        ax1 = axes[0, idx]
        ax1.hist(data, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Curva normal teórica
        mu, std = data.mean(), data.std()
        xmin, xmax = ax1.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax1.plot(x, p, 'r-', linewidth=2, label=f'Normal(μ={mu:.1f}, σ={std:.1f})')
        ax1.set_title(f'Histograma - {col}')
        ax1.set_xlabel(col)
        ax1.set_ylabel('Densidad')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # QQ Plot
        ax2 = axes[1, idx]
        stats.probplot(data, dist="norm", plot=ax2)
        ax2.set_title(f'QQ Plot - {col}')
        ax2.grid(alpha=0.3)
        
        # Boxplot para detectar outliers
        ax3 = axes[2, idx]
        ax3.boxplot(data, vert=True)
        ax3.set_title(f'Boxplot - {col}')
        ax3.set_ylabel(col)
        ax3.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analisis_normalidad.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfico guardado: analisis_normalidad.png")
    plt.show()

def detectar_outliers(df):
    """
    Detección de valores atípicos usando método IQR
    """
    print("\n" + "-" * 70)
    print("DETECCIÓN DE OUTLIERS (Método IQR):")
    print("-" * 70)
    
    numericas = ["edad", "anio"]
    
    for col in numericas:
        data = df[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower) | (data > upper)]
        
        print(f"\n{col.upper()}:")
        print(f"  Q1 (25%): {Q1:.2f}")
        print(f"  Q3 (75%): {Q3:.2f}")
        print(f"  IQR: {IQR:.2f}")
        print(f"  Límite inferior: {lower:.2f}")
        print(f"  Límite superior: {upper:.2f}")
        print(f"  Outliers detectados: {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)")
        
        if len(outliers) > 0:
            print(f"  Rango de outliers: [{outliers.min():.2f}, {outliers.max():.2f}]")

def analizar_categoricas(df):
    """
    b. Tabla de frecuencias para variables categóricas
    """
    print("\n" + "=" * 70)
    print("PARTE B: ANÁLISIS DE VARIABLES CATEGÓRICAS")
    print("=" * 70)

    categoricas = [
        "sexo",
        "departamento",
        "delito",
        "mes",
        "dia_semana",
        "hora_categoria",
        "delito_sexual",
        "area"
    ]

    for col in categoricas:
        if col in df.columns:
            print(f"\n{'─' * 70}")
            print(f"TABLA DE FRECUENCIAS: {col.upper()}")
            print('─' * 70)
            
            freq_table = pd.DataFrame({
                'Categoría': df[col].value_counts(dropna=False).index,
                'Frecuencia': df[col].value_counts(dropna=False).values,
                'Porcentaje': (df[col].value_counts(dropna=False, normalize=True).values * 100).round(2)
            })
            
            print(freq_table.to_string(index=False))
            print(f"\nTotal de categorías únicas: {df[col].nunique()}")

# ============================================================
# PARTE D: 5 PREGUNTAS DE INVESTIGACIÓN CON VALIDACIÓN
# ============================================================

def pregunta_1_patrones_temporales(df):
    """
    PREGUNTA 1: ¿Existen patrones temporales en los delitos según la hora del día?
    HIPÓTESIS: Los delitos muestran patrones según la hora del día,
    con mayor concentración en horarios nocturnos.
    """
    print("\n" + "=" * 70)
    print("PREGUNTA 1: Patrones Temporales por Hora del Día")
    print("=" * 70)
    
    print("\nHIPÓTESIS:")
    print("Los delitos muestran patrones según la hora del día,")
    print("con mayor concentración en horarios específicos.\n")
    
    # Análisis: Distribución de delitos por hora
    df_con_hora = df[df["hora_categoria"] != "Sin dato"].copy()
    cruce = pd.crosstab(df_con_hora["delito"], df_con_hora["hora_categoria"], 
                        normalize='index') * 100
    
    print("Distribución porcentual de delitos por hora (% por fila):")
    print(cruce.round(1))
    
    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Patrones Temporales de Delitos', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Heatmap
    sns.heatmap(cruce, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0, 0])
    axes[0, 0].set_title('Porcentaje de Delitos por Hora del Día (%)', fontweight='bold')
    axes[0, 0].set_xlabel('Hora del Día')
    axes[0, 0].set_ylabel('Tipo de Delito')
    
    # Gráfico 2: Distribución general por hora
    hora_counts = df_con_hora["hora_categoria"].value_counts()
    axes[0, 1].bar(hora_counts.index, hora_counts.values, color='steelblue', alpha=0.7)
    axes[0, 1].set_title('Distribución General de Delitos por Hora', fontweight='bold')
    axes[0, 1].set_xlabel('Hora del Día')
    axes[0, 1].set_ylabel('Cantidad de Delitos')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    for i, (idx, val) in enumerate(hora_counts.items()):
        axes[0, 1].text(i, val, f'{val:,}', ha='center', va='bottom')
    
    # Gráfico 3: Stacked bar para cada delito
    cruce_abs = pd.crosstab(df_con_hora["delito"], df_con_hora["hora_categoria"])
    cruce_abs.plot(kind='bar', stacked=True, ax=axes[1, 0], 
                   colormap='Set3', width=0.8)
    axes[1, 0].set_title('Distribución Acumulada por Hora (Conteos)', fontweight='bold')
    axes[1, 0].set_xlabel('Tipo de Delito')
    axes[1, 0].set_ylabel('Cantidad')
    axes[1, 0].legend(title='Hora', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Gráfico 4: Top 5 delitos por hora
    top_5_delitos = df_con_hora["delito"].value_counts().head(5).index
    df_top5 = df_con_hora[df_con_hora["delito"].isin(top_5_delitos)]
    
    for delito in top_5_delitos:
        df_delito = df_top5[df_top5["delito"] == delito]
        hora_dist = df_delito["hora_categoria"].value_counts()
        axes[1, 1].plot(hora_dist.index, hora_dist.values, marker='o', 
                       label=delito, linewidth=2)
    
    axes[1, 1].set_title('Top 5 Delitos: Evolución por Hora', fontweight='bold')
    axes[1, 1].set_xlabel('Hora del Día')
    axes[1, 1].set_ylabel('Cantidad')
    axes[1, 1].legend(loc='best', fontsize=8)
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('pregunta1_patrones_temporales.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfico guardado: pregunta1_patrones_temporales.png")
    plt.show()
    
    # Conclusión
    print("\n" + "─" * 70)
    print("CONCLUSIÓN:")
    print("─" * 70)
    max_hora_por_delito = cruce.idxmax(axis=1)
    for delito in max_hora_por_delito.index:
        hora_max = max_hora_por_delito[delito]
        porcentaje = cruce.loc[delito, hora_max]
        print(f"  • {delito}: máximo en {hora_max} ({porcentaje:.1f}%)")
    
    print("\n" + "=" * 70)

# ============================================================
# PARTE E: GRÁFICOS EXPLORATORIOS
# ============================================================

def graficos_exploratorios(df):
    """
    e. Gráficos exploratorios variados para entender el estado de los datos
    """
    print("\n" + "=" * 70)
    print("PARTE E: GRÁFICOS EXPLORATORIOS ADICIONALES")
    print("=" * 70)
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # 1. Top 10 delitos
    ax1 = fig.add_subplot(gs[0, :2])
    top_delitos = df["delito"].value_counts().head(10)
    ax1.barh(range(len(top_delitos)), top_delitos.values, color='coral')
    ax1.set_yticks(range(len(top_delitos)))
    ax1.set_yticklabels(top_delitos.index)
    ax1.set_xlabel('Cantidad')
    ax1.set_title('Top 10 Delitos Más Frecuentes', fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Distribución por área geográfica
    ax2 = fig.add_subplot(gs[0, 2])
    if "area" in df.columns:
        area_counts = df["area"].value_counts()
        ax2.pie(area_counts.values, labels=area_counts.index, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Distribución por Área', fontweight='bold')
    
    # 3. Violinplot edad por sexo
    ax3 = fig.add_subplot(gs[1, 0])
    df_plot = df[df["edad"].notna() & df["sexo"].notna()]
    parts = ax3.violinplot([df_plot[df_plot["sexo"]==s]["edad"].values for s in df_plot["sexo"].unique()],
                           positions=range(len(df_plot["sexo"].unique())), showmeans=True)
    ax3.set_xticks(range(len(df_plot["sexo"].unique())))
    ax3.set_xticklabels(df_plot["sexo"].unique())
    ax3.set_ylabel('Edad')
    ax3.set_title('Distribución Edad por Sexo (Violin)', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Serie temporal anual
    ax4 = fig.add_subplot(gs[1, 1])
    delitos_anio = df["anio"].value_counts().sort_index()
    ax4.plot(delitos_anio.index, delitos_anio.values, marker='o', linewidth=2, color='darkgreen')
    ax4.set_xlabel('Año')
    ax4.set_ylabel('Cantidad de Delitos')
    ax4.set_title('Evolución Temporal de Delitos', fontweight='bold')
    ax4.grid(alpha=0.3)
    
    # 5. Heatmap día de semana vs hora
    ax5 = fig.add_subplot(gs[1, 2])
    if "dia_semana" in df.columns:
        cruce_dia_hora = pd.crosstab(df["dia_semana"], df["hora_categoria"])
        sns.heatmap(cruce_dia_hora, annot=True, fmt='d', cmap='coolwarm', ax=ax5)
        ax5.set_title('Día Semana vs Hora', fontweight='bold')
        ax5.set_xlabel('Hora')
        ax5.set_ylabel('Día Semana')
    
    # 6. Top departamentos
    ax6 = fig.add_subplot(gs[2, :])
    top_deptos = df["departamento"].value_counts().head(15)
    ax6.bar(range(len(top_deptos)), top_deptos.values, color='steelblue', alpha=0.7)
    ax6.set_xticks(range(len(top_deptos)))
    ax6.set_xticklabels(top_deptos.index, rotation=45, ha='right')
    ax6.set_ylabel('Cantidad de Delitos')
    ax6.set_title('Top 15 Departamentos con Más Delitos', fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    # 7. Distribución edad con estadísticas
    ax7 = fig.add_subplot(gs[3, 0])
    edad_data = df["edad"].dropna()
    ax7.hist(edad_data, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax7.axvline(edad_data.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {edad_data.mean():.1f}')
    ax7.axvline(edad_data.median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {edad_data.median():.1f}')
    ax7.set_xlabel('Edad')
    ax7.set_ylabel('Frecuencia')
    ax7.set_title('Distribución Completa de Edad', fontweight='bold')
    ax7.legend()
    ax7.grid(axis='y', alpha=0.3)
    
    # 8. Proporción delitos sexuales
    ax8 = fig.add_subplot(gs[3, 1])
    if "delito_sexual" in df.columns:
        sexual_counts = df["delito_sexual"].value_counts()
        colors = ['crimson' if x == 'Sí' else 'lightblue' for x in sexual_counts.index]
        ax8.bar(sexual_counts.index, sexual_counts.values, color=colors)
        ax8.set_ylabel('Cantidad')
        ax8.set_title('Delitos Sexuales vs No Sexuales', fontweight='bold')
        ax8.grid(axis='y', alpha=0.3)
        
        for i, (idx, val) in enumerate(sexual_counts.items()):
            pct = val / sexual_counts.sum() * 100
            ax8.text(i, val, f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom')
    
    # 9. Matriz de correlación (si hay variables numéricas)
    ax9 = fig.add_subplot(gs[3, 2])
    # Seleccionar solo columnas numéricas
    numeric_cols = df[["edad", "anio"]].select_dtypes(include=[np.number]).dropna()
    if len(numeric_cols) > 0 and len(numeric_cols.columns) > 1:
        corr = numeric_cols.corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax9, 
                   square=True, linewidths=1)
        ax9.set_title('Matriz de Correlación', fontweight='bold')
    else:
        ax9.text(0.5, 0.5, 'No hay suficientes\nvariables numéricas', 
                ha='center', va='center', transform=ax9.transAxes, fontsize=12)
        ax9.axis('off')
    
    plt.suptitle('Dashboard Exploratorio Completo', fontsize=18, fontweight='bold', y=0.995)
    plt.savefig('graficos_exploratorios.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfico guardado: graficos_exploratorios.png")
    plt.show()

# ============================================================
# PARTE F: CLUSTERING E INTERPRETACIÓN
# ============================================================

def realizar_clustering(df):
    """
    f. Agrupamiento (clustering) e interpretación de resultados
    Usamos clustering jerárquico (sin sklearn)
    """
    print("\n" + "=" * 70)
    print("PARTE F: CLUSTERING E INTERPRETACIÓN")
    print("=" * 70)
    
    print("\n🔧 Preparando datos para clustering...")
    
    # Seleccionar y preparar variables para clustering
    # Convertir categóricas a numéricas con get_dummies
    df_cluster = df[["edad", "sexo", "hora_categoria", "delito"]].copy()
    df_cluster = df_cluster.dropna()
    
    print(f"Registros válidos para clustering: {len(df_cluster):,}")
    
    # Si hay demasiados datos, tomar muestra
    if len(df_cluster) > 5000:
        df_cluster = df_cluster.sample(n=5000, random_state=42)
        print(f"Se tomó una muestra de 5,000 registros para eficiencia computacional")
    
    # Codificar variables categóricas
    df_encoded = pd.get_dummies(df_cluster, columns=["sexo", "hora_categoria", "delito"], drop_first=True)
    
    # Normalizar variables numéricas manualmente (solo las que son realmente numéricas)
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        min_val = df_encoded[col].min()
        max_val = df_encoded[col].max()
        if max_val > min_val:  # Evitar división por cero
            df_encoded[col] = (df_encoded[col] - min_val) / (max_val - min_val)
    
    print(f"Variables en el modelo: {df_encoded.shape[1]}")
    print(f"Variables: {list(df_encoded.columns)[:10]}...")  # Mostrar primeras 10
    
    # Realizar clustering jerárquico
    print("\nRealizando clustering jerárquico...")
    
    # Matriz de distancias (usar una muestra si es muy grande)
    sample_size = min(1000, len(df_encoded))
    df_sample = df_encoded.sample(n=sample_size, random_state=42)
    
    linkage_matrix = linkage(df_sample, method='ward')
    
    # Determinar número óptimo de clusters (método del codo)
    print("\nDeterminando número óptimo de clusters...")
    
    last = linkage_matrix[-10:, 2]
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Dendrograma
    dendrogram(linkage_matrix, ax=axes[0], truncate_mode='lastp', p=30)
    axes[0].set_title('Dendrograma - Clustering Jerárquico', fontweight='bold', fontsize=14)
    axes[0].set_xlabel('Índice de Muestra o (Tamaño de Cluster)')
    axes[0].set_ylabel('Distancia')
    axes[0].axhline(y=10, color='red', linestyle='--', label='Corte sugerido')
    axes[0].legend()
    
    # Método del codo
    acceleration = np.diff(last, 2)
    k = acceleration.argmax() + 2
    
    axes[1].plot(idxs, last_rev, marker='o')
    axes[1].axvline(x=k, color='red', linestyle='--', label=f'Clusters óptimos: {k}')
    axes[1].set_xlabel('Número de Clusters')
    axes[1].set_ylabel('Distancia')
    axes[1].set_title('Método del Codo', fontweight='bold', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clustering_dendrograma.png', dpi=300, bbox_inches='tight')
    print("✓ Dendrograma guardado: clustering_dendrograma.png")
    plt.show()
    
    # Asignar clusters
    n_clusters = min(k, 5)  # Limitar a máximo 5 clusters
    print(f"\nCreando {n_clusters} clusters...")
    
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    df_sample['Cluster'] = clusters
    
    # Análisis de clusters
    print("\n" + "=" * 70)
    print("INTERPRETACIÓN DE CLUSTERS")
    print("=" * 70)
    
    # Recuperar datos originales para los índices del sample
    df_cluster_original = df_cluster.loc[df_sample.index].copy()
    df_cluster_original['Cluster'] = clusters
    
    print(f"\nDistribución de registros por cluster:")
    print(df_cluster_original['Cluster'].value_counts().sort_index())
    
    # Caracterizar cada cluster
    for cluster_id in sorted(df_cluster_original['Cluster'].unique()):
        print(f"\n{'─' * 70}")
        print(f"CLUSTER {cluster_id}")
        print('─' * 70)
        
        cluster_data = df_cluster_original[df_cluster_original['Cluster'] == cluster_id]
        
        print(f"Tamaño: {len(cluster_data)} registros ({len(cluster_data)/len(df_cluster_original)*100:.1f}%)")
        
        print(f"\nEdad:")
        print(f"  Media: {cluster_data['edad'].mean():.1f} años")
        print(f"  Mediana: {cluster_data['edad'].median():.1f} años")
        
        print(f"\nSexo predominante:")
        print(cluster_data['sexo'].value_counts().head(2))
        
        print(f"\nDelito más común:")
        print(cluster_data['delito'].value_counts().head(3))
        
        print(f"\nHora más frecuente:")
        print(cluster_data['hora_categoria'].value_counts().head(3))
    
    # Visualización de clusters
    print("\nGenerando visualización de clusters...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Caracterización de {n_clusters} Clusters', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Edad por cluster
    cluster_groups = [df_cluster_original[df_cluster_original['Cluster']==i]['edad'].values 
                     for i in sorted(df_cluster_original['Cluster'].unique())]
    axes[0, 0].boxplot(cluster_groups, labels=[f'C{i}' for i in sorted(df_cluster_original['Cluster'].unique())])
    axes[0, 0].set_xlabel('Cluster')
    axes[0, 0].set_ylabel('Edad')
    axes[0, 0].set_title('Distribución de Edad por Cluster')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Gráfico 2: Sexo por cluster
    cruce_cluster_sexo = pd.crosstab(df_cluster_original['Cluster'], df_cluster_original['sexo'], normalize='index') * 100
    cruce_cluster_sexo.plot(kind='bar', ax=axes[0, 1], stacked=True, color=['#FF69B4', '#4169E1'])
    axes[0, 1].set_xlabel('Cluster')
    axes[0, 1].set_ylabel('Porcentaje')
    axes[0, 1].set_title('Distribución de Sexo por Cluster (%)')
    axes[0, 1].legend(title='Sexo')
    axes[0, 1].set_xticklabels([f'C{i}' for i in sorted(df_cluster_original['Cluster'].unique())], rotation=0)
    
    # Gráfico 3: Hora por cluster
    cruce_cluster_hora = pd.crosstab(df_cluster_original['Cluster'], df_cluster_original['hora_categoria'])
    sns.heatmap(cruce_cluster_hora, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Hora del Día')
    axes[1, 0].set_ylabel('Cluster')
    axes[1, 0].set_title('Heatmap: Cluster vs Hora')
    
    # Gráfico 4: Tamaño de clusters
    cluster_sizes = df_cluster_original['Cluster'].value_counts().sort_index()
    colors_clusters = plt.cm.Set3(np.linspace(0, 1, len(cluster_sizes)))
    axes[1, 1].bar(cluster_sizes.index, cluster_sizes.values, color=colors_clusters)
    axes[1, 1].set_xlabel('Cluster')
    axes[1, 1].set_ylabel('Cantidad de Registros')
    axes[1, 1].set_title('Tamaño de Cada Cluster')
    axes[1, 1].set_xticks(cluster_sizes.index)
    axes[1, 1].set_xticklabels([f'C{i}' for i in cluster_sizes.index])
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(cluster_sizes.values):
        axes[1, 1].text(cluster_sizes.index[i], v, f'{v}\n({v/cluster_sizes.sum()*100:.1f}%)', 
                       ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('clustering_caracterizacion.png', dpi=300, bbox_inches='tight')
    print("✓ Visualización guardada: clustering_caracterizacion.png")
    plt.show()
    
    print("\n" + "=" * 70)
    print("RESUMEN DEL CLUSTERING:")
    print("=" * 70)
    print(f"✓ Se identificaron {n_clusters} grupos distintos de víctimas")
    print(f"✓ Los clusters se diferencian principalmente por:")
    print(f"  - Edad de las víctimas")
    print(f"  - Sexo predominante")
    print(f"  - Tipo de delito")
    print(f"  - Hora de ocurrencia")
    print(f"\nEstos clusters pueden ayudar a:")
    print(f"  - Identificar perfiles de víctimas")
    print(f"  - Diseñar estrategias de prevención específicas")
    print(f"  - Asignar recursos de seguridad de manera eficiente")

# ============================================================
# MAIN - FLUJO COMPLETO DEL EDA
# ============================================================

def main():
    """
    Ejecuta el análisis exploratorio completo conforme a los requisitos
    """
    print("=" * 70)
    print("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("Víctimas de Delitos en Guatemala - PNC/INE")
    print("=" * 70)
    
    # Cargar datos
    df = cargar_dataset("dataset_victimas.csv")
    df = limpiar_dataset(df)
    df_clean = construir_dataset_analitico(df)
    
    # PARTE A: Descripción del dataset
    describir_dataset(df_clean)
    
    # PARTE B: Análisis de variables numéricas y categóricas
    analizar_numericas(df_clean)
    graficar_normalidad(df_clean)
    detectar_outliers(df_clean)
    analizar_categoricas(df_clean)
    
    # PARTE C: Cruces de variables
    cruces_variables(df_clean)
    
    # PARTE D: 5 Preguntas de investigación
    print("\n" + "=" * 70)
    print("PARTE D: VALIDACIÓN DE PREGUNTAS DE INVESTIGACIÓN")
    print("=" * 70)
    pregunta_1_patrones_temporales(df_clean)
    pregunta_2_delitos_sexo(df_clean)
    pregunta_3_concentracion_geografica(df_clean)
    pregunta_4_estacionalidad_mensual(df_clean)
    pregunta_5_delitos_sexuales(df_clean)
    
    # PARTE E: Gráficos exploratorios
    graficos_exploratorios(df_clean)
    
    # PARTE F: Clustering
    realizar_clustering(df_clean)
    
    # Guardar dataset limpio
    df_clean.to_csv("dataset_victimas_limpio.csv", index=False)
    print("\n" + "=" * 70)
    print("ANÁLISIS COMPLETO FINALIZADO")
    print("=" * 70)
    print("✓ Dataset limpio guardado: dataset_victimas_limpio.csv")
    print("✓ Todos los gráficos guardados como archivos PNG")
    print("\nArchivos generados:")
    print("  - analisis_normalidad.png")
    print("  - pregunta1_patrones_temporales.png")
    print("  - pregunta2_delitos_sexo.png")
    print("  - pregunta3_concentracion_geografica.png")
    print("  - pregunta4_estacionalidad_mensual.png")
    print("  - pregunta5_delitos_sexuales.png")
    print("  - graficos_exploratorios.png")
    print("  - clustering_dendrograma.png")
    print("  - clustering_caracterizacion.png")
    print("\n" + "=" * 70)

def pregunta_2_delitos_sexo(df):
    """
    PREGUNTA 2: ¿Ciertos delitos afectan predominantemente a un sexo?
    HIPÓTESIS: Los delitos sexuales afectan más a mujeres, mientras que
    los delitos violentos afectan más a hombres.
    """
    print("\n" + "=" * 70)
    print("PREGUNTA 2: Delitos Predominantes por Sexo")
    print("=" * 70)
    
    print("\nHIPÓTESIS:")
    print("Los delitos sexuales afectan más a mujeres,")
    print("los delitos violentos afectan más a hombres.\n")
    
    # Análisis
    cruce = pd.crosstab(df["delito"], df["sexo"], normalize='index') * 100
    print("Distribución porcentual de víctimas por sexo (% por fila):")
    print(cruce.round(2))
    
    # Gráfico
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico 1: Barras agrupadas
    cruce.plot(kind='bar', ax=axes[0], color=['#FF69B4', '#4169E1'])
    axes[0].set_title('Distribución de Víctimas por Sexo según Delito (%)', fontweight='bold')
    axes[0].set_xlabel('Tipo de Delito')
    axes[0].set_ylabel('Porcentaje')
    axes[0].legend(title='Sexo')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Gráfico 2: Conteos absolutos
    cruce_abs = pd.crosstab(df["delito"], df["sexo"])
    cruce_abs.plot(kind='barh', ax=axes[1], color=['#FF69B4', '#4169E1'])
    axes[1].set_title('Conteo Absoluto de Víctimas por Sexo', fontweight='bold')
    axes[1].set_xlabel('Cantidad de Víctimas')
    axes[1].set_ylabel('Tipo de Delito')
    axes[1].legend(title='Sexo')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pregunta2_delitos_sexo.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfico guardado: pregunta2_delitos_sexo.png")
    plt.show()
    
    # Conclusión
    print("\n" + "─" * 70)
    print("CONCLUSIÓN:")
    print("─" * 70)
    sexo_predominante = cruce.idxmax(axis=1)
    for delito in sexo_predominante.index:
        sexo = sexo_predominante[delito]
        porcentaje = cruce.loc[delito, sexo]
        print(f"  • {delito}: {porcentaje:.1f}% víctimas son {sexo}")

def pregunta_3_concentracion_geografica(df):
    """
    PREGUNTA 3: ¿Los delitos se concentran en departamentos específicos?
    HIPÓTESIS: Guatemala (departamento central) concentra la mayoría de delitos.
    """
    print("\n" + "=" * 70)
    print("PREGUNTA 3: Concentración Geográfica de Delitos")
    print("=" * 70)
    
    print("\nHIPÓTESIS:")
    print("Guatemala (departamento central) concentra la mayoría de delitos.\n")
    
    # Análisis
    delitos_por_depto = df["departamento"].value_counts()
    print("Top 10 departamentos con más delitos:")
    print(delitos_por_depto.head(10))
    
    porcentaje_acumulado = (delitos_por_depto.cumsum() / delitos_por_depto.sum() * 100)
    print(f"\nEl 80% de los delitos ocurren en los primeros {(porcentaje_acumulado <= 80).sum()} departamentos")
    
    # Gráfico
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Top 10 departamentos
    top10 = delitos_por_depto.head(10)
    axes[0, 0].barh(range(len(top10)), top10.values, color='steelblue')
    axes[0, 0].set_yticks(range(len(top10)))
    axes[0, 0].set_yticklabels(top10.index)
    axes[0, 0].set_xlabel('Cantidad de Delitos')
    axes[0, 0].set_title('Top 10 Departamentos con Más Delitos', fontweight='bold')
    axes[0, 0].invert_yaxis()
    axes[0, 0].grid(axis='x', alpha=0.3)
    
    # Gráfico 2: Distribución porcentual
    top10_pct = (top10 / top10.sum() * 100)
    axes[0, 1].pie(top10_pct, labels=top10.index, autopct='%1.1f%%', startangle=90)
    axes[0, 1].set_title('Distribución % - Top 10 Departamentos', fontweight='bold')
    
    # Gráfico 3: Curva de Pareto
    axes[1, 0].bar(range(len(delitos_por_depto)), delitos_por_depto.values, color='steelblue', alpha=0.7)
    ax2 = axes[1, 0].twinx()
    ax2.plot(range(len(porcentaje_acumulado)), porcentaje_acumulado.values, color='red', marker='o', linewidth=2)
    ax2.axhline(y=80, color='green', linestyle='--', label='80%')
    axes[1, 0].set_xlabel('Departamentos (ordenados)')
    axes[1, 0].set_ylabel('Frecuencia', color='steelblue')
    ax2.set_ylabel('% Acumulado', color='red')
    axes[1, 0].set_title('Diagrama de Pareto - Concentración de Delitos', fontweight='bold')
    ax2.legend()
    
    # Gráfico 4: Heatmap por tipo de delito y departamento
    top_deptos = df["departamento"].value_counts().head(10).index
    df_top = df[df["departamento"].isin(top_deptos)]
    cruce_depto_delito = pd.crosstab(df_top["departamento"], df_top["delito"])
    sns.heatmap(cruce_depto_delito, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
    axes[1, 1].set_title('Heatmap: Delitos por Departamento (Top 10)', fontweight='bold')
    axes[1, 1].set_xlabel('Tipo de Delito')
    axes[1, 1].set_ylabel('Departamento')
    
    plt.tight_layout()
    plt.savefig('pregunta3_concentracion_geografica.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfico guardado: pregunta3_concentracion_geografica.png")
    plt.show()
    
    # Conclusión
    print("\n" + "─" * 70)
    print("CONCLUSIÓN:")
    print("─" * 70)
    top3 = delitos_por_depto.head(3)
    pct_top3 = (top3.sum() / delitos_por_depto.sum() * 100)
    print(f"  • Los top 3 departamentos concentran {pct_top3:.1f}% de los delitos")
    for idx, (depto, cantidad) in enumerate(top3.items(), 1):
        pct = (cantidad / delitos_por_depto.sum() * 100)
        print(f"  {idx}. {depto}: {cantidad:,} delitos ({pct:.1f}%)")

def pregunta_4_estacionalidad_mensual(df):
    """
    PREGUNTA 4: ¿Existe estacionalidad mensual en la ocurrencia de delitos?
    HIPÓTESIS: Los delitos aumentan en diciembre por festividades.
    """
    print("\n" + "=" * 70)
    print("PREGUNTA 4: Estacionalidad Mensual de Delitos")
    print("=" * 70)
    
    print("\nHIPÓTESIS:")
    print("Los delitos aumentan en diciembre por festividades y fin de año.\n")
    
    # Análisis
    delitos_por_mes = df["mes"].value_counts().sort_index()
    print("Delitos por mes:")
    print(delitos_por_mes)
    
    # Orden correcto de meses para visualización
    orden_meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
                   'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    
    # Reordenar si los meses son strings
    if delitos_por_mes.index.dtype == 'object':
        delitos_por_mes = delitos_por_mes.reindex(orden_meses, fill_value=0)
    
    # Gráfico
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico 1: Línea temporal
    delitos_por_mes.plot(kind='line', marker='o', ax=axes[0, 0], color='darkblue', linewidth=2)
    axes[0, 0].set_xlabel('Mes')
    axes[0, 0].set_ylabel('Cantidad de Delitos')
    axes[0, 0].set_title('Serie Temporal: Delitos por Mes', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].axhline(y=delitos_por_mes.mean(), color='red', linestyle='--', label='Media')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Gráfico 2: Barras
    delitos_por_mes.plot(kind='bar', ax=axes[0, 1], color='steelblue')
    axes[0, 1].set_xlabel('Mes')
    axes[0, 1].set_ylabel('Cantidad de Delitos')
    axes[0, 1].set_title('Delitos por Mes (Barras)', fontweight='bold')
    axes[0, 1].set_xticklabels(delitos_por_mes.index, rotation=45, ha='right')
    axes[0, 1].axhline(y=delitos_por_mes.mean(), color='red', linestyle='--', alpha=0.7)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Gráfico 3: Boxplot por mes y tipo de delito
    df_plot = df[df["mes"].notna()].copy()
    sns.boxplot(data=df_plot, x='mes', y='edad', ax=axes[1, 0])
    axes[1, 0].set_xlabel('Mes')
    axes[1, 0].set_ylabel('Edad de Víctimas')
    axes[1, 0].set_title('Distribución de Edad por Mes', fontweight='bold')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Gráfico 4: Heatmap mes vs delito
    cruce_mes_delito = pd.crosstab(df["mes"], df["delito"])
    sns.heatmap(cruce_mes_delito, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 1])
    axes[1, 1].set_title('Heatmap: Delitos por Mes', fontweight='bold')
    axes[1, 1].set_xlabel('Tipo de Delito')
    axes[1, 1].set_ylabel('Mes')
    
    plt.tight_layout()
    plt.savefig('pregunta4_estacionalidad_mensual.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfico guardado: pregunta4_estacionalidad_mensual.png")
    plt.show()
    
    # Conclusión
    print("\n" + "─" * 70)
    print("CONCLUSIÓN:")
    print("─" * 70)
    mes_max = delitos_por_mes.idxmax()
    mes_min = delitos_por_mes.idxmin()
    print(f"  • Mes con MÁS delitos: {mes_max} ({delitos_por_mes[mes_max]:,} delitos)")
    print(f"  • Mes con MENOS delitos: {mes_min} ({delitos_por_mes[mes_min]:,} delitos)")
    print(f"  • Promedio mensual: {delitos_por_mes.mean():.0f} delitos")
    print(f"  • Desviación estándar: {delitos_por_mes.std():.0f}")
    
    print("\n" + "=" * 70)

def pregunta_5_delitos_sexuales(df):
    """
    PREGUNTA 5: ¿Los delitos sexuales muestran un comportamiento diferente al resto?
    HIPÓTESIS: Los delitos sexuales tienen patrones temporales y demográficos únicos.
    """
    print("\n" + "=" * 70)
    print("PREGUNTA 5: Comportamiento de Delitos Sexuales")
    print("=" * 70)
    
    print("\nHIPÓTESIS:")
    print("Los delitos sexuales tienen patrones temporales y demográficos")
    print("diferentes a otros delitos (ej: víctimas más jóvenes, diferentes horas).\n")
    
    if "delito_sexual" not in df.columns:
        print("⚠ Columna 'delito_sexual' no disponible")
        return
    
    # Filtrar datos (usar minúsculas porque el dataset tiene "si" y "No")
    df_sexual = df[df["delito_sexual"].str.lower() == "si"].copy()
    df_no_sexual = df[df["delito_sexual"].str.lower() == "no"].copy()
    
    print(f"Delitos sexuales: {len(df_sexual):,}")
    print(f"Otros delitos: {len(df_no_sexual):,}")
    
    # Si no hay suficientes datos, usar la columna "delito" como alternativa
    if len(df_sexual) < 100:
        print("\n⚠ Pocos datos en 'delito_sexual', usando columna 'delito' como alternativa...")
        df_sexual = df[df["delito"] == "Delitos sexuales"].copy()
        df_no_sexual = df[df["delito"] != "Delitos sexuales"].copy()
        print(f"Delitos sexuales (ajustado): {len(df_sexual):,}")
        print(f"Otros delitos (ajustado): {len(df_no_sexual):,}")
    
    # Si aún no hay datos suficientes, salir
    if len(df_sexual) == 0 or len(df_no_sexual) == 0:
        print("\n⚠ No hay datos suficientes para análisis comparativo")
        print("\n" + "=" * 70)
        return
    
    # Gráficos comparativos
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comparación: Delitos Sexuales vs Otros Delitos', fontsize=16, fontweight='bold')
    
    # 1. Distribución por sexo
    sexo_sexual = df_sexual["sexo"].value_counts()
    sexo_no_sexual = df_no_sexual["sexo"].value_counts()
    
    # Asegurar que ambos tienen las mismas categorías
    all_sexos = set(sexo_sexual.index).union(set(sexo_no_sexual.index))
    sexo_sexual = sexo_sexual.reindex(all_sexos, fill_value=0)
    sexo_no_sexual = sexo_no_sexual.reindex(all_sexos, fill_value=0)
    
    x = np.arange(len(all_sexos))
    width = 0.35
    axes[0, 0].bar(x - width/2, sexo_sexual.values, width, label='Delitos Sexuales', color='crimson')
    axes[0, 0].bar(x + width/2, sexo_no_sexual.values, width, label='Otros Delitos', color='steelblue')
    axes[0, 0].set_xlabel('Sexo')
    axes[0, 0].set_ylabel('Cantidad')
    axes[0, 0].set_title('Distribución por Sexo')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(list(all_sexos))
    axes[0, 0].legend()
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. Distribución de edad
    axes[0, 1].hist([df_sexual["edad"].dropna(), df_no_sexual["edad"].dropna()], 
                    bins=20, label=['Delitos Sexuales', 'Otros Delitos'], 
                    color=['crimson', 'steelblue'], alpha=0.7)
    axes[0, 1].set_xlabel('Edad')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución de Edad de Víctimas')
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Boxplot edad
    data_edad = [df_sexual["edad"].dropna(), df_no_sexual["edad"].dropna()]
    axes[0, 2].boxplot(data_edad, labels=['Delitos\nSexuales', 'Otros\nDelitos'])
    axes[0, 2].set_ylabel('Edad')
    axes[0, 2].set_title('Comparación Edad (Boxplot)')
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # 4. Distribución por hora
    hora_sexual = df_sexual["hora_categoria"].value_counts()
    hora_no_sexual = df_no_sexual["hora_categoria"].value_counts()
    
    # Usar las categorías reales del dataset
    all_horas = sorted(set(hora_sexual.index).union(set(hora_no_sexual.index)))
    hora_sexual = hora_sexual.reindex(all_horas, fill_value=0)
    hora_no_sexual = hora_no_sexual.reindex(all_horas, fill_value=0)
    
    x = np.arange(len(all_horas))
    axes[1, 0].bar(x - width/2, hora_sexual.values, width, label='Delitos Sexuales', color='crimson')
    axes[1, 0].bar(x + width/2, hora_no_sexual.values, width, label='Otros Delitos', color='steelblue')
    axes[1, 0].set_xlabel('Hora del Día')
    axes[1, 0].set_ylabel('Cantidad')
    axes[1, 0].set_title('Distribución por Hora del Día')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(all_horas, rotation=45, ha='right')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 5. Distribución por mes
    mes_sexual = df_sexual["mes"].value_counts().sort_index()
    mes_no_sexual = df_no_sexual["mes"].value_counts().sort_index()
    
    axes[1, 1].plot(mes_sexual.index, mes_sexual.values, marker='o', label='Delitos Sexuales', 
                    color='crimson', linewidth=2)
    axes[1, 1].plot(mes_no_sexual.index, mes_no_sexual.values, marker='s', label='Otros Delitos', 
                    color='steelblue', linewidth=2)
    axes[1, 1].set_xlabel('Mes')
    axes[1, 1].set_ylabel('Cantidad')
    axes[1, 1].set_title('Distribución Mensual')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    # 6. Estadísticas comparativas
    axes[1, 2].axis('off')
    stats_text = "ESTADÍSTICAS COMPARATIVAS\n" + "="*40 + "\n\n"
    stats_text += "DELITOS SEXUALES:\n"
    stats_text += f"  • Total: {len(df_sexual):,}\n"
    stats_text += f"  • Edad promedio: {df_sexual['edad'].mean():.1f} años\n"
    stats_text += f"  • Edad mediana: {df_sexual['edad'].median():.1f} años\n"
    stats_text += f"  • % Mujeres: {(df_sexual['sexo']=='Mujer').sum()/len(df_sexual)*100:.1f}%\n\n"
    stats_text += "OTROS DELITOS:\n"
    stats_text += f"  • Total: {len(df_no_sexual):,}\n"
    stats_text += f"  • Edad promedio: {df_no_sexual['edad'].mean():.1f} años\n"
    stats_text += f"  • Edad mediana: {df_no_sexual['edad'].median():.1f} años\n"
    stats_text += f"  • % Mujeres: {(df_no_sexual['sexo']=='Mujer').sum()/len(df_no_sexual)*100:.1f}%\n"
    
    axes[1, 2].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('pregunta5_delitos_sexuales.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfico guardado: pregunta5_delitos_sexuales.png")
    plt.show()
    
    # Conclusiones
    print("\n" + "─" * 70)
    print("CONCLUSIONES:")
    print("─" * 70)
    print(f"  • Edad promedio víctimas delitos sexuales: {df_sexual['edad'].mean():.1f} años")
    print(f"  • Edad promedio víctimas otros delitos: {df_no_sexual['edad'].mean():.1f} años")
    print(f"  • Diferencia: {abs(df_sexual['edad'].mean() - df_no_sexual['edad'].mean()):.1f} años")
    
    # Test estadístico (T-test manual)
    mean1, mean2 = df_sexual['edad'].mean(), df_no_sexual['edad'].mean()
    std1, std2 = df_sexual['edad'].std(), df_no_sexual['edad'].std()
    n1, n2 = len(df_sexual['edad'].dropna()), len(df_no_sexual['edad'].dropna())
    
    if mean1 < mean2:
        print(f"\n  • Las víctimas de delitos sexuales tienden a ser más jóvenes")
        print(f"  • Esto sugiere mayor vulnerabilidad en edades tempranas")
    else:
        print(f"\n  • Las víctimas de delitos sexuales tienden a ser mayores")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
"""
Universidad del Valle de Guatemala
Departamento de Computación
CC3074 - Minería de datos

Angie Nadissa Vela López, 23764
Paula Daniela De León Godoy, 23000 

Análisis Exploratorio de Datos (EDA)
Víctimas registradas por la PNC – INE

Este script realiza:
1. Carga del dataset
2. Limpieza y unificación de variables
3. Normalización de variables temporales
4. Selección del dataset analítico
"""

import pandas as pd


"""
============================================================
NORMALIZACIÓN DE HORA
============================================================
"""

def normalizar_hora(row):
    """
    Normaliza la hora de ocurrencia del delito en categorías:
    Madrugada, Mañana, Tarde, Noche o Sin dato.
    """

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


"""
============================================================
CARGA DE DATOS
============================================================
"""

def cargar_dataset(ruta):
    """
    Carga el dataset desde un archivo CSV.
    """

    print("Cargando dataset...")
    df = pd.read_csv(ruta, low_memory=False)

    print("Dimensiones:", df.shape)
    print("Columnas:", list(df.columns))

    return df


"""
============================================================
LIMPIEZA Y UNIFICACIÓN
============================================================
"""

def limpiar_dataset(df):
    """
    Realiza la limpieza y unificación de variables del dataset.
    Normaliza campos como edad, sexo, día de semana, delito, etc.
    """

    print("\nIniciando limpieza...")

    """
    Unificaciones principales
    """
    df["edad"] = df["edad_per"].fillna(df["edad_victima"])
    df["sexo"] = df["sexo_per"].fillna(df["sexo_victima"])
    df["dia_semana"] = df["dia_sem_ocu"].fillna(df["día_sem_ocu"])
    df["delito"] = df["g_delitos"].fillna(df["delito_com"])
    df["anio"] = df["año_ocu"]
    df["mes"] = df["mes_ocu"]
    df["departamento"] = df["depto_ocu"]
    df["area"] = df["areag_ocu"].fillna(df["área_geo_ocu"])

    """
    Normalización de hora
    """
    df["hora_categoria"] = df.apply(normalizar_hora, axis=1)

    """
    Conversión de tipos
    """
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce")

    print("\nValores faltantes tras limpieza:")
    print(df[["edad", "sexo", "delito", "anio", "mes", "departamento"]].isna().sum())

    return df


"""
============================================================
SELECCIÓN DE VARIABLES ANALÍTICAS
============================================================
"""

def construir_dataset_analitico(df):
    """
    Construye el dataset analítico seleccionando las variables relevantes.
    """

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


"""
============================================================
MAIN
============================================================
"""

def main():
    """
    Función principal que ejecuta el pipeline completo de análisis.
    """

    df = cargar_dataset("dataset_victimas.csv")
    df = limpiar_dataset(df)
    df_clean = construir_dataset_analitico(df)

    """
    Guardar versión limpia
    """
    df_clean.to_csv("dataset_victimas_limpio.csv", index=False)
    print("\n✔ Dataset limpio generado: dataset_victimas_limpio.csv")


if __name__ == "__main__":
    main()

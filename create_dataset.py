import pandas as pd
import glob
import re
import os

ruta = "data_ine/*.sav"   # Ajusta carpeta
archivos = sorted(glob.glob(ruta))

dataframes = []
reporte = []

total_filas_originales = 0
todas_columnas = set()

for archivo in archivos:
    
    nombre = os.path.basename(archivo)
    
    match = re.search(r"(20\d{2})", nombre)
    year_file = int(match.group(1)) if match else None
    try:
        print(f"Leyendo: {archivo}")
        df = pd.read_spss(archivo)
        df.columns = df.columns.str.lower()
        
        if "año_ocu" not in df.columns or df["año_ocu"].isna().all():
            df["año_ocu"] = year_file
        
        filas, columnas = df.shape
        total_filas_originales += filas
        todas_columnas.update(df.columns)

        reporte.append({
            "archivo": archivo,
            "filas": filas,
            "columnas": columnas,
            "nombres_columnas": list(df.columns)
        })

        dataframes.append(df)

    except Exception as e:
        print(f"Error leyendo {archivo}: {e}")

if not dataframes:
    print("No se pudo leer ningún archivo.")
    exit()

# Unir todo
df_final = pd.concat(dataframes, ignore_index=True, sort=True)

print("\n==============================")
print("VERIFICACIÓN DE UNIÓN")
print("==============================")

print(f"Filas esperadas: {total_filas_originales}")
print(f"Filas obtenidas: {df_final.shape[0]}")

if total_filas_originales == df_final.shape[0]:
    print("Todas las filas se unieron correctamente.")
else:
    print("Diferencia en filas. Revisar archivos.")

print(f"Total de columnas finales: {df_final.shape[1]}")

# Verificar columnas faltantes por archivo
print("\n==============================")
print("REVISIÓN DE COLUMNAS")
print("==============================")

columnas_finales = set(df_final.columns)

for r in reporte:
    columnas_archivo = set(r["nombres_columnas"])
    faltantes = columnas_finales - columnas_archivo

    if faltantes:
        print(f"\n{r['archivo']} no contiene:")
        print(faltantes)
    else:
        print(f"\n{r['archivo']} contiene todas las columnas.")

# Guardar dataset
df_final.to_csv("dataset_victimas.csv", index=False)
print("\nArchivo generado: dataset_victimas.csv")

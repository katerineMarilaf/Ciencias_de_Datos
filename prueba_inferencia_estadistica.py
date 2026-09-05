import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import t

np.random.seed(42)  # Semilla para reproducibilidad

#=====================================================================================================
#                           1. Carga y exploración de datos
#=====================================================================================================

# Simulación de DataFrame con 200 registros

n = 200

datos = {
    "Edad": np.random.randint(18, 50, size=n),
    "Genero": np.random.choice(["Femenino", "Masculino", "Otro"], size=n, p=[0.48, 0.48, 0.04]),
    "Puntaje_Satisfaccion": np.random.randint(1, 11, size=n),
    "Horas_Estudio_Semanales": np.round(np.random.normal(loc=12, scale=4, size=n), 1)
}

df = pd.DataFrame(datos)

# Estadísticas descriptivas básicas de variables numéricas
print("\n---ESTADÍSTICAS DESCRIPTIVAS---\n")
print(df.describe())
print("\n---Estadísticas descriptivas básicas---\n")
print(df.describe().T[["mean", "50%", "std"]].rename(columns={"50%": "mediana"}))

# Verificación de valores nulos
print("\n--- Conteo de Valores Nulos ---\n")
print(df.isnull().sum())

#No existen valores nulos. Si hubiesen existido valores nulos se podría eliminar los registros si fuesen pocos;
# reemplazar valores por la media, mediana ó moda; investigar la causa de los datos faltantes antes de decidir cómo tratarlos.

#=====================================================================================================
#                           2. Distribución y visualización
#=====================================================================================================

plt.figure(figsize=(8,5))

plt.hist(df["Puntaje_Satisfaccion"],
         bins=10,
         color="#BDD7EE",
         edgecolor="black")

plt.title("Histograma del puntaje de satisfacción")
plt.xlabel("Puntaje")
plt.ylabel("Frecuencia")
plt.show()

media = np.mean(df["Puntaje_Satisfaccion"])
varianza = np.var(df["Puntaje_Satisfaccion"])

print("\n---Media y Varianza utilizando Numpy---\n")
print(f"Media: {media:.2f}")
print(f"Varianza: {varianza:.2f}")

#Interpreta brevemente si los datos parecen seguir una distribución normal:

#El histograma no presenta una forma de campana característica de una distribución normal. 
#Los puntajes de satisfacción se distribuyen de forma mas o menos uniforme entre los valores posibles, 
#por lo que no parecen seguir una distribución normal.


#=====================================================================================================
#                           3. Intervalo de confianza
#=====================================================================================================

desviacion = np.std(df["Puntaje_Satisfaccion"], ddof=1)
t_critico= stats.t.ppf(0.975, df=n - 1)
margen_error = t_critico * (desviacion / np.sqrt(n))
limite_inferior = media - margen_error
limite_superior = media + margen_error

print("\n--- INTERVALO DE CONFIANZA AL 95%---\n")
print(f"t critico: {t_critico:.2f}")
print(f"Media: {media:.2f}")
print(f"Desviación estándar: {desviacion:.2f}")
print(f"Intervalo de Confianza al 95%: [{limite_inferior:.2f}, {limite_superior:.2f}]")
print(f"Interpretación: Con un 95% de nivel de confianza, la media poblacional de satisfacción se encuentra entre {limite_inferior:.2f} y {limite_superior:.2f}.\n")

# Explica en tus palabras qué significa este intervalo en el contexto del estudio:

# Esto significa que existe un 95% de confianza de que la verdadera media
# poblacional de satisfacción de todos los estudiantes se encuentre dentro de este intervalo [5.20, 6.05].

#=====================================================================================================
#                           4. Prueba de hipótesis
#=====================================================================================================

#H0= El puntaje promedio de satisfacción es igual a 7
mu0 = 7

t_stat = (media - mu0) / (desviacion / np.sqrt(n))

p_value = 2 * (1 - t.cdf(abs(t_stat), df=n-1))

print(f"t = {t_stat:.2f}, p-valor = {p_value:.4f}")

alpha = 0.05
print("\n--- PRUEBA DE HIPÓTESIS BILATERAL---\n")

print("H₀=El puntaje promedio de satisfacción es igual a 7\n")
print(f"H₀: μ = {mu0}  |  H₁: μ ≠ {mu0}")
print()
if p_value <= alpha:
    print(f"p ({p_value:.4f}) ≤ α ({alpha})  →  Se RECHAZA H₀")
    print("\n Conclusión:Existe evidencia estadística suficiente para concluir que el puntaje promedio de satisfacción es diferente de 7.\n")
else:
    print(f"p ({p_value:.4f}) > α ({alpha})  →  No se rechaza H₀")
    print("\nConclusión: Existe evidencia estadística suficiente para concluir que el puntaje promedio de satisfacción es igual a 7\n")

#Explica la interpretación del resultado en términos prácticos para la empresa:

#Los resultados indican que el puntaje promedio de satisfacción de los estudiantes 
#es significativamente diferente de 7. Dado que el promedio obtenido (5.62) es inferior a ese valor, 
#la empresa podría revisar y mejorar aspectos del curso para aumentar la satisfacción de los estudiantes.

#=====================================================================================================
#                           5. Reflexión final
#=====================================================================================================

#La estadística inferencial permite obtener conclusiones sobre una población a partir del análisis de una muestra. 
#Los intervalos de confianza y las pruebas de hipótesis ayudan a tomar decisiones basadas en evidencia,
#disminuyendo la incertidumbre y facilitando la planificación de mejoras en los procesos y servicios ofrecidos por la empresa.


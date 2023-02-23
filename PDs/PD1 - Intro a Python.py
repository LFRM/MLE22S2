# # Práctica dirigida 1

# ## 1. Introducción a Python

# <center><img src="https://www.freecodecamp.org/news/content/images/2021/08/chris-ried-ieic5Tq8YMk-unsplash.jpg" alt="drawing" width="550"/></center>
# 
# Python es un lenguaje de programación con las siguientes características:
# * **Popular**. Es bastante utilizado y es por eso que existe una gran variedad de librerías ([¿Qué tanto es utilizado?](https://survey.stackoverflow.co/2022/#section-most-popular-technologies-programming-scripting-and-markup-languages)).
# * **Legible**. Su sintaxis es fácil de entender ([¿Qué tan fácil de entender?](https://survey.stackoverflow.co/2022/#most-loved-dreaded-and-wanted-language-want)).
# * **Versátil**. Puede ser utilizado en Desarrollo de Software o Ciencia de Datos ([¿Qué otros usos tiene Python?](https://www.jetbrains.com/lp/devecosystem-2021/python/#Python_what-do-you-use-python-for)).
# 
# Por lo tanto, Python será nuestra principal herramienta para aplicar Machine Learning (ML).
# 
# > *Fun fact. Guido van Rossum, el creador de Python, le puso ese nombre al lenguaje de programación por el programa de TV de la BBC, Monty Python’s Flying Circus. Él es un gran fan.*
# 
# Antes de aplicar ML, primero veamos algunas funcionalidades básicas: operadores matemáticos, estructuras de datos y estructuras de control.

# ### 1.1. Operadores matemáticos

# In[ ]:


# Variables
a = 27
b = 5

# Operaciones
sum_ = a + b
diff = a - b
prod = a * b
div = a / b
int_div = a // b
mod = a % b
exp = a ** b


# In[ ]:


# Output usando formatter
print("Resultados")
print(f"Suma: {sum_} \t\t\t Tipo de dato: {type(sum_)}")
print(f"Resta: {diff} \t\t\t Tipo de dato: {type(diff)}")
print(f"Multiplicación: {prod} \t\t Tipo de dato: {type(prod)}")
print(f"División: {div} \t\t\t Tipo de dato {type(div)}")
print(f"División entera: {int_div} \t\t Tipo de dato {type(int_div)}")
print(f"Módulo: {mod} \t\t\t Tipo de dato {type(mod)}")
print(f"Exponenciación: {exp} \t Tipo de dato {type(exp)}")


# ### 1.2. Estructuras de datos

# In[ ]:


# Variables
a = [10, 7 , 3, 8, 2, "Hola"] # Lista
b = (10, 7 , 3, 8, 2, "Hola") # Tupla
c = {10, 7 , 3, 8, 2, "Hola"} # Set
d = {"Pos1": 10, "Pos2": 7, "Pos3": 3, "Pos4": 8, "Pos5": 2, "Pos6": "Hola"} # Diccionario


# In[ ]:


# Output
print("Lista")
print("Valor:", a)
print("Longitud:", len(a))
print()

print("Tupla")
print("Valor:", b)
print("Longitud:", len(b))
print()

print("Set")
print("Valor:", c)
print("Longitud:", len(c))
print()

print("Diccionario")
print("Valor:", d)
print("Longitud:", len(d))
print()


# ### 1.3. Estructuras de control

# In[ ]:


# Estructura condicional
if d["Pos4"] == 9:
  print("El key 4 del diccionario contiene un 8")
elif d["Pos5"] == 2:
  print("El key 5 del diccionario contiene un 2")


# In[ ]:


# Estructura iterativa usando FOR
for i in c:
  print(i, end=" ")


# In[ ]:


# Estructura iterativa usando WHILE
count = 0
while count < len(a):
  if a[count] == "Hola":
    print(f"¡Hay un intruso en la posicion {count} de la lista!")
    break
  count += 1


# ### 1.4. Practicando

# Para hacer los ejercicios ir al siguiente [link](https://codecollab.io/@roniepaolo/PD1) (editar cuando se den las indicaciones) y resolver:
# 
# 
# 1. Declara las variables `a, b, c`, donde `a != 0` y resuelve la ecuacion general cuadrática $\frac {-b \pm \sqrt {b^2 - 4ac}}{2a}$
# 
# 2. Declara la lista `x = [10, 20, -10, -20, 100, 30, -100]` y halla la suma de todos los elementos positivos utilizando las estructuras de control FOR e IF.
# 
# 3. Declara la lista `x = [1, 15, 20, 33, 23, 76, 100]` y usando la estructura de control FOR e IF imprime si el número es par o impar.
# 
# **Reto:** Resolver el siguiente [contest](https://www.hackerrank.com/mle-introduccion-a-python)

# In[ ]:


a = 1
b = 2
c = 3
raizpos = (-b + (b ** 2 - 4 * a * c) ** 0.5) / 2 * a
raizneg = (-b - (b ** 2 - 4 * a * c) ** 0.5) / 2 * a
print(raizpos)
print(raizneg)


# In[ ]:


x = [10, 20, -1, -20, 100, 30, -100]
sumcu = 0
for i in range(len(x)):
  if x[i] > 0:
    sumcu = sumcu + x[i]
print(sumcu)


# In[ ]:


x = [10, 20, -1, -20, 100, 30, -100]
sumcu = 0
for i in x:
    if i > 0:
      sumcu = sumcu + i
print(sumcu)


# # 2. Análisis de datos

# El siguiente dataset contiene las canciones más escuchadas en Perú (Spotify) en la primera mitad del 2020.
# 
# |Columnas|Tipo|Descripción|
# |---|---|---|
# |album|string|El álbum de la canción.|
# |artist|string|El artista de la canción.|
# |track_name|string|Nombre de la canción.|
# |track_energy|float|La energía es una medida de 0,0 a 1,0 y representa una medida perceptiva de intensidad y actividad. Por lo general, las pistas enérgicas se sienten rápidas, fuertes y ruidosas. Por ejemplo, el death metal tiene mucha energía, mientras que un preludio de Bach tiene una puntuación baja en la escala. Las características perceptivas que contribuyen a este atributo incluyen el rango dinámico, el volumen percibido, el timbre, la tasa de inicio y la entropía general.|
# |track_loudness|float|El volumen general de una pista en decibelios (dB). Los valores de sonoridad se promedian en toda la pista y son útiles para comparar la sonoridad relativa de las pistas. El volumen es la cualidad de un sonido que es el principal correlato psicológico de la fuerza física (amplitud). Los valores típicos oscilan entre -60 y 0 db.|

# ### 2.1. Librerías

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


# ### 2.2. Lectura de los datos

# In[ ]:


# Se descarga el dataset
get_ipython().system('curl -L -O https://github.com/roniepaolo/spotify-charts-analysis/raw/master/data/dataset_tracks_peru.csv')


# In[ ]:


# Se lee el dataset
df = pd.read_csv("dataset_tracks_peru.csv", \
  usecols=["album", "artist", "track_name", "track_energy", "track_loudness"])


# In[ ]:


# Se imprimen las primeras 5 filas del dataset
df.head()


# In[ ]:


# Tipos de datos
df.dtypes


# ### 2.3. Análisis exploratorio de los datos

# In[ ]:


# Estadística descriptiva del dataset
df.describe()


# In[ ]:


# Se imprime el tamaño del dataset
print("El tamaño del dataset es:", df.shape)
print(df.shape[0], "filas y", df.shape[1], "columnas")


# In[ ]:


df["artist"].value_counts()[:7]


# In[ ]:


# Piechart
df["artist"].value_counts()[:7].plot.pie(figsize=(10, 6), autopct="%.1f%%")
plt.show()


# In[ ]:


df.loc[:, ["track_name","track_energy"]].sort_values(by="track_energy", ascending=False, ignore_index=True).iloc[:5, :]


# In[ ]:


# Bar plot
df.loc[:, ["track_name","track_energy"]]. \
  sort_values(by="track_energy", ascending=False, ignore_index=True). \
  iloc[:5, :]. \
  plot.bar("track_name", "track_energy", figsize=(10, 6))
plt.show()


# In[ ]:


# Histogram
df[["track_loudness", "track_energy"]].plot.hist(alpha=0.5, figsize=(10, 6), bins=15)
plt.show()


# In[ ]:


# Boxplot
df.boxplot(["track_loudness", "track_energy"], figsize=(10, 6))
plt.show()


# In[ ]:


# Scatterplot
df.plot.scatter("track_loudness", "track_energy", figsize=(10, 6))
plt.show()


# ### 2.4. Modelamiento

# In[ ]:


# Se crean las variables X (input) y y (output)
X = df[["track_loudness"]]
y = df["track_energy"]


# In[ ]:


# Regresión lineal simple
reg = LinearRegression()
reg.fit(X, y)


# In[ ]:


# Predicción
y_pred = reg.predict(X)


# In[ ]:


# Regresión y métricas
print(f"Regresión: y = {reg.intercept_:.4f} + {reg.coef_[0]:.4f} * x")
print(f"MAE: {mean_absolute_error(y, y_pred):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y, y_pred) * 100:.4f} %")


# In[ ]:


# Scatterplot
df.plot.scatter("track_loudness", "track_energy", figsize=(10, 6))
plt.plot(X, reg.intercept_ + reg.coef_[0] * X, color="red", alpha=0.5)
plt.show()


# ### 2.5. Practicando

# Esperar a que se muestre el código del Kahoot.

# ### 2.6 Videos recomendados

# * [IBM Cognitive Class - Python for Data Science](https://cognitiveclass.ai/courses/python-for-data-science)
# * [IBM Cognitive Class - Data Analysis](https://cognitiveclass.ai/courses/data-analysis-python)
# * [Socratica - Introduction to Python](https://www.youtube.com/watch?v=bY6m6_IIN94&list=PLi01XoE8jYohWFPpC17Z-wWhPOSuh8Er-)
# * [Amigoscode - Introduction to Python](https://www.youtube.com/watch?v=mJEpimi_tFo)

#!/usr/bin/env python
# coding: utf-8

# # PD4

# Los métodos de reducción de dimensionalidad son importantes cuando se tienen datasets con un gran número de features que podrían estar altamente correlacionadas. En este contexto, la estimación por MCO podría hacer overfitting de la data de entreno y la predicción tendría un MSE en prueba excesivamente alto. Por ello, se justifica crear un nuevo set de features que sean $m$ combinaciones lineales de los $p$ features iniciales, donde $m \ll p$. Estos métodos introducen un grado de sesgo en la estimación de los parámetros a favor de una mucho menor varianza en la predicción.
# 
# Se le proporciona la base de datos Hitters. Esta contiene datos de 322 jugadores de las Grandes Ligas de Béisbol en las temporadas de 1986 y 1987. La base muestra diferentes atributos de su desempeño y experiencia profesional, así como su salario anual en el comienzo de la temporada en miles de dólares. Se le pide crear un modelo que prediga el salario anual de los beisbolistas teniendo en cuenta sus atributos y desempeño. Para ello, haga lo siguiente: 
# 
# 1) Cargue los datos y elabore el preprocesamiento de la base\
# 2) Realice el análisis exploratorio\
# 3) Estime un modelo de regresión lineal para predecir el salario de los jugadores.\
# 4) Estime un modelo de regresión por componentes principales.\
# 5) Compare los resultados entre los dos modelos, tanto del desempeño en train como en test.
# 
# |Columnas|Tipo|Descripción|
# |---|---|---|
# |AtBat|Continua |Número de veces al bate en 1986|
# |Hits|Continua |Número de hits en 1986|
# |HmRun|Continua|Número de jonrones en 1986|
# |Runs|Continua |Número de carreras en 1986|
# |RBI|Continua |Número de carreras impulsadas en 1986|
# |Walks|Continua |Número de paseos en 1986|
# |Years|Continua |Número de años en las ligas mayores|
# |CAtBat|Continua |Número de veces al bate durante su carrera|
# |CHits|Continua |Número de hits durante su carrera|
# |CHmRun|Continua|Número de jonrones durante su carrera|
# |CRuns|Continua |Número de carreras durante su carrera|
# |CRBI|Continua |Número de carreras impulsadas durante su carrera|
# |CRuns| Continua|Número de bases por bolas durante su carrera|
# |Liga| Categórica/Ordinal|Un factor con niveles entre A y N que indican la liga del jugador a fines de 1986|
# |División|Categórica/Ordinal |Un factor con niveles entre E y W que indican la división del jugador a fines de 1986|
# |PutOuts|Continua |Número de salidas en 1986|
# |Assists|Continua |Número de asistencias en 1986|
# |Errors|Continua |Número de errores en 1986|
# |Salary|Continua |Salario anual de 1987 el día de la inauguración en miles de dólares|
# |NewLeague|Categórica/Ordinal |Un factor con niveles A y N que indican la liga del jugador a principios de 1987|

# # 1. Análisis de datos

# ## 1.1. Librerías

# In[3]:


# Se importan las librerías que se van a utilizar
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import seaborn as sns
from scipy.stats import pearsonr

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error

pd.set_option("display.max_columns", None)


# ## 1.2. Lectura de datos

# In[ ]:


# Se descarga la data
get_ipython().system('curl -L -O https://github.com/JWarmenhoven/ISLR-python/raw/master/Notebooks/Data/Hitters.csv')


# In[ ]:


# Se lee en un dataframe
df_hitters = pd.read_csv("Hitters.csv")


# In[ ]:


# Se imprimen 5 filas de forma aleatoria
df_hitters.sample(5)


# In[ ]:


# Se imprime el tamaño del dataset
print("El tamaño del dataset es:", df_hitters.shape)
print(df_hitters.shape[0], "filas y", df_hitters.shape[1], "columnas")


# ## 1.3. Preprocesamiento

# In[ ]:


# Se verifica si hay nulos
df_hitters.isnull().sum()


# In[ ]:


# Se verifican los tipos de variables
df_hitters.dtypes


# In[ ]:


# Se elimina la columna Unnamed: 0
df_hitters.drop("Unnamed: 0", axis=1, inplace=True)


# In[ ]:


df_hitters["Salary"]


# In[ ]:


# Se eliminan los valores nulos del target
df_hitters.dropna(axis=0, subset=["Salary"], inplace=True)

# Cada vez que se eliminan las filas, procurar hacerle un reset a los indices
df_hitters.reset_index(drop=True, inplace=True)


# In[ ]:


# Verificar los valores de las variables categoricas
print(df_hitters["League"].value_counts(), end="\n\n")
print(df_hitters["Division"].value_counts(), end="\n\n")
print(df_hitters["NewLeague"].value_counts())


# In[ ]:


# Se convierten las variables categoricas a dummies
df_hitters = pd.get_dummies(df_hitters, \
  columns=["League", "Division", "NewLeague"], \
  prefix=["League", "Division", "NewLeague"], \
  drop_first=True)


# In[ ]:


df_hitters.head()


# In[ ]:


# Se imprime el tamaño del dataset
print("El tamaño del dataset es:", df_hitters.shape)
print(df_hitters.shape[0], "filas y", df_hitters.shape[1], "columnas")


# ## 1.4. Análisis exploratorio de los datos

# In[ ]:


# Estadistica descriptiva
round(df_hitters.describe(), 2)


# In[ ]:


# Se grafica un histograma para ver la distribucion del salario con 15 bins
df_hitters["Salary"].plot.hist(bins=15, figsize=(10, 6))
plt.show()


# In[ ]:


# Matriz de correlacion en forma de mapa de calor
plt.figure(figsize=(14, 8))
corr_matrix = round(df_hitters.corr(), 2)
mask = np.triu(np.ones_like(corr_matrix))
sns.heatmap(corr_matrix, annot=True, mask=mask)
plt.show()


# ## 1.5. Separacion de conjunto de datos

# In[ ]:


# Se generan los conjuntos de datos de entrenamiento y prueba
X = df_hitters.drop("Salary", axis=1)
y = df_hitters["Salary"]

X_train, X_test, y_train, y_test = \
  train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Se estandariza la data
scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)


# # 2. Regresion Lineal

# In[ ]:


# Ajustar el modelo a los datos
lm = LinearRegression()
lm.fit(X_train_scaled, y_train)
y_pred = lm.predict(X_test_scaled)


# In[ ]:


# Metricas de desempenho del modelo
print(f"Correlación: {pearsonr(y_test, y_pred)[0]:.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.show()


# # 3. Regresion por Compenentes Principales

# In[ ]:


# Se generan todos los componentes principales
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# In[ ]:


# Graficar el Scree Plot
plt.figure(figsize=(10, 6))
plt.bar(range(1, X_train_pca.shape[1] + 1), pca.explained_variance_ratio_)
plt.plot(range(1, X_train_pca.shape[1] + 1), np.cumsum(pca.explained_variance_ratio_), marker="s")
plt.xticks(range(1, X_train_pca.shape[1] + 1))
plt.title("Scree Plot")
plt.xlabel("Componentes Principales")
plt.ylabel("Ratio de Varianza Explicada")
plt.show()


# In[ ]:


# Se aplica la regresion a los componentes principales usando cross validation
maes = []
for i in range(1, X_train_pca.shape[1] + 1):
  pcr = LinearRegression()
  cv = LeaveOneOut()
  scores = cross_validate(pcr, X_train_pca[:, :i], y_train, cv=cv, scoring="neg_mean_absolute_error")
  maes.append(np.array(abs(scores["test_score"])).mean())


# In[ ]:


# Se grafica el MAE por cada componente
plt.figure(figsize=(10, 6))
plt.plot(range(1, X_train_pca.shape[1] + 1), maes)
plt.xticks(range(1, X_train_pca.shape[1] + 1))
plt.title("Desempenho por Componentes")
plt.xlabel("Componentes")
plt.ylabel("MAE")
plt.show()


# In[ ]:


# Se aplica PCR con 3 componentes
pcr = LinearRegression()
pcr.fit(X_train_pca[:, :4], y_train)
y_pred = pcr.predict(X_test_pca[:, :4])


# In[ ]:


# Metricas de desempenho del modelo
print(f"Correlación: {pearsonr(y_test, y_pred)[0]:.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.show()


# # 4. Conclusion

# La regresión lineal que utiliza todos los predictores originales nos da un mejor resultado predictivo que la regresión por componentes principales con 4 componentes. La regresión lineal nos arroja un MAE de 211.12, y una correlación entre valores predichos y realizados de 56.23%.

# # 5. Ejercicio

# Genere 10000 observaciones de una distribución normal bivariada con media [0,0]
# y matriz de covarianzas [[1, theta], [theta, 1]]
# ​
#  1. Para theta = 0.5\
#  a) Calcule los dos componentes principales de X\
#  b) ¿Qué porcentaje de la varianza total explica el primer componente principal? 
#  2. Grafique las observaciones y superponga\
#  a) Los vectores propios\
#  b) La estimación de la función de regresión x2 = beta x1 + epsilon\
#  c) Repita a) y b) con diferentes theta = 0.5 y theta = -0.5. ¿Qué cambia?
# ​

# In[33]:


# Generación de data
n = 10000
mu = [0, 0]


# In[34]:


# Para theta = 0.5
theta = 0.5
Sigma = [[1, theta], [theta, 1]]
x1, x2 = np.random.default_rng(seed=42).multivariate_normal(mu, Sigma, n).T
X = np.column_stack((x1, x2))
XTX = np.matmul(X.T, X) / (n - 1)


# In[35]:


# a) Calcule los dos componentes principales de X
theta = 0.5
l, V = LA.eig(XTX)
#P = np.matmul(X, V)


# In[ ]:


# b) ¿Qué porcentaje de la varianza total explica el primer componente principal?
print("Primer ratio de varianza explicada:", l[0]/(l[0]+l[1]))
print("Segundo ratio de varianza explicada:", l[1]/(l[0]+l[1]))


# In[ ]:


# Grafique las observaciones y superponga
# a) Los vectores propios
# b) La estimación de la función de regresión x2 = beta x1 + epsilon
# regresion lineal
m_rl = np.cov(x1, x2)[0][1]
m_v1 = V[1][0] / V[0][0]
m_v2 = V[1][1] / V[0][1]

plt.plot(x1, x2, "x")
plt.axis("equal")
plt.plot(x1, m_rl * x1, label="Regresion Lineal")
plt.plot(x1, m_v1 * x1, label="Componente 1")
plt.plot(x1, m_v2 * x1, label="Componente 2")
plt.legend(loc="upper left")
plt.show()


# In[ ]:


# c) Repita a) y b) con diferentes theta = 0.5 y theta = -0.5. ¿Qué cambia?

# Cuando theta = 0.5 el vector propio asociado al valor propio más alto es paralelo al vector (1,1). 
# Cuando theta = -0.5 el vector propio asociado al valor propio más alto es paralelo al vector (1,-1).


#!/usr/bin/env python
# coding: utf-8

# # PD 5

# Como últimos métodos para lidiar conjuntos de datos de dimensión alta revisaremos las Regresiones Lasso y Ridge. Al aplicar estos métodos, veremos que algunos coeficientes estimados pueden tomar el valor de hasta exactamente cero (Lasso), siendo estas variables las que deberíamos descartar, ya que no aportan poder predictivo.
# 
# Se le proporciona la base de datos Hitters. Esta contiene datos de 322 jugadores de las Grandes Ligas de Béisbol en las temporadas de 1986 y 1987. La base muestra diferentes atributos de su desempeño y experiencia profesional, así como su salario anual en el comienzo de la temporada en miles de dólares. Se le pide crear un modelo que prediga el salario anual de los beisbolistas teniendo en cuenta sus atributos y desempeño.
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

# In[ ]:


# Se importan las librerías que se van a utilizar
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import seaborn as sns
from scipy.stats import pearsonr

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_validate, GridSearchCV
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, Ridge, Lasso
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
lm_ans_corr = pearsonr(y_test, y_pred)[0]
lm_ans_mae = mean_absolute_error(y_test, y_pred)
print(f"Correlación: {lm_ans_corr:.4f}")
print(f"MAE: {lm_ans_mae:.4f}")
print()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.show()


# In[ ]:


# Visualizo los pesos de cada variable
pd.DataFrame({"Variables": X_train.columns, "Pesos": lm.coef_}). \
  sort_values(by="Pesos", ascending=False, key=lambda x: abs(x)). \
  plot.bar(x="Variables", y="Pesos", figsize=(10, 6))
plt.show()


# # 3. Regresión Ridge

# In[ ]:


# Se evalúan varias valores de lambda
ridge = RidgeCV(alphas=np.arange(1, 400, 2), cv=10)
ridge.fit(X_train_scaled, y_train)
y_pred = ridge.predict(X_test_scaled)


# In[ ]:


# Metricas de desempenho del modelo
ridge_ans_corr = pearsonr(y_test, y_pred)[0]
ridge_ans_mae = mean_absolute_error(y_test, y_pred)
ridge_ans_lambda = ridge.alpha_
print(f"Correlación: {ridge_ans_corr:.4f}")
print(f"MAE: {ridge_ans_mae:.4f}")
print(f"Mejor lambda: {ridge_ans_lambda:.4f}")
print()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.show()


# In[ ]:


# Visualizo los pesos de cada variable
pd.DataFrame({"Variables": X_train.columns, "Pesos": ridge.coef_}). \
  sort_values(by="Pesos", ascending=False, key=lambda x: abs(x)). \
  plot.bar(x="Variables", y="Pesos", figsize=(10, 6))
plt.show()


# # 4. Regresión Lasso

# In[ ]:


# Se evalúan varias valores de lambda
lasso = LassoCV(alphas=np.arange(0.1, 10, 0.1), cv=10, max_iter=5000)
lasso.fit(X_train_scaled, y_train)
y_pred = lasso.predict(X_test_scaled)


# In[ ]:


# Metricas de desempenho del modelo
lasso_ans_corr = pearsonr(y_test, y_pred)[0]
lasso_ans_mae = mean_absolute_error(y_test, y_pred)
lasso_ans_lambda = ridge.alpha_
print(f"Correlación: {lasso_ans_corr:.4f}")
print(f"MAE: {lasso_ans_mae:.4f}")
print(f"Mejor lambda: {lasso_ans_lambda:.4f}")
print()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.show()


# In[ ]:


# Visualizo los pesos de cada variable
pd.DataFrame({"Variables": X_train.columns, "Pesos": lasso.coef_}). \
  sort_values(by="Pesos", ascending=False, key=lambda x: abs(x)). \
  plot.bar(x="Variables", y="Pesos", figsize=(10, 6))
plt.show()


# # 6. Conclusion

# In[ ]:


df_ans = pd.DataFrame({"LinearRegression": [lm_ans_corr, lm_ans_mae], \
  "Ridge": [ridge_ans_corr, ridge_ans_mae], \
  "Lasso": [lasso_ans_corr, lasso_ans_mae]})
df_ans.index = ["Correlacion", "Mae"]
df_ans


# In[ ]:





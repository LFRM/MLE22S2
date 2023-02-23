#!/usr/bin/env python
# coding: utf-8

# # Práctica dirigida 2

# El siguiente dataset contiene los datos de publicidad de las ventas de un producto en 200 tiendas diferentes, junto con los presupuestos de publicidad para tres medios distintos: televisión, radio y periódicos.
# 
# |Columnas|Tipo|Descripción|
# |---|---|---|
# |TV|float|Presupuesto de publicidad del producto para televisión (\$).|
# |Radio|float|Presupuesto de publicidad del producto para radio (\$).|
# |Newspaper|float|Presupuesto de publicidad del producto para periódicos (\$).|
# |Sales|float|Unidades vendidas del producto (unid.).|

# ## 1. Análisis de datos

# Realice un análisis de datos:  <br />
# a) Elimine columnas innecesarias.  <br />
# b) Analice gráficamente la distribución de los datos a través de box plots y scatter plots.  <br />
# c) Resuelva las observaciones nulas y outliers en caso existan.  <br />
# 
# 

# ### 1.1 Librerías

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from scipy.stats import pearsonr


# ### 1.2. Lectura de datos

# In[ ]:


# Se descarga el dataset
get_ipython().system('curl -L -O https://github.com/JWarmenhoven/ISLR-python/raw/master/Notebooks/Data/Advertising.csv')


# In[ ]:


# Se lee el dataset
df = pd.read_csv("Advertising.csv")


# In[ ]:


# Se imprimen 5 filas de forma aleatoria
df.sample(5)


# In[ ]:


# Se imprime el tamaño del dataset
print("El tamaño del dataset es:", df.shape)
print(df.shape[0], "filas y", df.shape[1], "columnas")


# ### 1.3. Preprocesamiento

# In[ ]:


# Tipo de datos de las variables
df.dtypes


# In[ ]:


# Variables nulas y no nulas
df.isnull().sum()


# In[ ]:


# Se eliminan las columnas que no se utilizaran
df.drop("Unnamed: 0", axis=1, inplace=True)


# ### 1.4. Análisis exploratorio de los datos

# In[ ]:


# Estadística descriptiva
df.describe()


# In[ ]:


# Visualizar varabilidad de los datos
df.boxplot(["TV", "Radio", "Newspaper"], figsize=(10, 6))
plt.show()


# In[ ]:


# Visualizar correlacion y distribucion
pd.plotting.scatter_matrix(df, alpha=0.5, figsize=(15, 10))
plt.show()


# In[ ]:


# Remover valores atipicos
q1 = df.describe()["Newspaper"]["25%"]
q3 = df.describe()["Newspaper"]["75%"]
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

df = df.loc[(df["Newspaper"] >= lower) & (df["Newspaper"] <= upper)]. \
  reset_index(drop=True)


# In[ ]:


# Visualizar varabilidad de los datos
df.boxplot(["TV", "Radio", "Newspaper"], figsize=(10, 6))
plt.show()


# In[ ]:


# Visualizar correlacion y distribucion
pd.plotting.scatter_matrix(df, alpha=0.5, figsize=(15, 10))
plt.show()


# ## 2. Regresión Lineal Simple

# Realice un análisis de regresión lineal simple:<br />
# a) Estime la regresión univariada de ventas en términos de uno de los gastos en publicidad. <br />
# b) Interprete los coeficientes de la regresión. <br />
# c) Interprete la significancia de los coeficientes, así como sus intervalos de confianza. <br />
# d) Interprete los estadísticos de bondad de ajuste de la regresión. <br />
# e) Calcule la distancia de Cook para cada observación. Comente. <br />
# f) Calcule el factor de inflación de varianza para cada variable. Comente. <br />

# ### 2.1. Inferencia

# In[ ]:


# Modelo de Regresión Lineal
lm = smf.ols("Sales ~ Newspaper", data=df)
res = lm.fit()

# Resumen de la respuesta
print(res.summary())


# ### 2.2. Distancia de Cook

# In[ ]:


# Se calcula la distancia de cook
influence = res.get_influence()
cooks = influence.cooks_distance


# In[ ]:


# Se grafica la distancia por cada observación
plt.figure(figsize=(10, 6))
plt.bar(df.index, cooks[0], label="Distancia de Cook")
#plt.axhline(1, linestyle="dashed", color="red", alpha=0.5, label="Limit")
plt.legend()
plt.show()


# ### 2.3. Factor de inflación de la varianza

# In[ ]:


# Se declara una función para el cálculo del VIF
def compute_vif(features):
  X = df.loc[:, features]
  X["intercept"] = 1
  vif = pd.DataFrame()
  vif["Variable"] = X.columns
  vif["VIF"] = \
    [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
  vif = vif[vif["Variable"] != "intercept"]
  return vif


# In[ ]:


# Se llama a la función para obtener la respuesta
compute_vif(df.drop(["Sales"], axis=1).columns).sort_values("VIF", ascending=False)


# ## 3. Regresión Lineal Múltiple

# Realice un análisis de regresión lineal simple: <br />
# a) Estime la regresión de ventas en términos de los tres tipos de gasto. <br />
# b) Interprete los coeficientes de la regresión. <br />
# c) Interprete la significancia de los coeficientes, así como sus intervalos de confianza. <br />
# d) Interprete los estadísticos de bondad de ajuste de la regresión. <br />
# e) Si alguna variable debe ser eliminada por ser poco significativa, hágalo y luego calcule el factor de inflación de varianza para cada variable. <br />
# f) Si algunas observaciones deben ser eliminadas por ser outliers, hágalo y luego realice un análisis predictivo a través de regresión lineal y genere estadísticos de desempeño para la predicción del modelo en la data de prueba.

# ### 3.1. Inferencia

# In[ ]:


# Modelo de Regresión Lineal
lm = smf.ols("Sales ~ Newspaper + Radio + TV", data=df)
res = lm.fit()


# In[ ]:


# Resumen de la respuesta
print(res.summary())


# ### 3.2. Se descarta Newspaper

# In[ ]:


# Modelo de Regresión Lineal
lm = smf.ols("Sales ~ Radio + TV", data=df)
res = lm.fit()


# In[ ]:


# Resumen de la respuesta
print(res.summary())


# ### 3.3. Factor de inflación de la varianza

# In[ ]:


# Se declara una función para el cálculo del VIF
def compute_vif(features):
  X = df.loc[:, features]
  X["intercept"] = 1
  vif = pd.DataFrame()
  vif["Variable"] = X.columns
  vif["VIF"] = \
    [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
  vif = vif[vif["Variable"] != "intercept"]
  return vif


# In[ ]:


# Se llama a la función para obtener la respuesta
compute_vif(df.drop(["Sales", "Newspaper"], axis=1).columns).sort_values("VIF", ascending=False)


# ### 3.4. Predicción

# In[ ]:


# Se separan conjuntos de datos de entrenamiento y prueba
X = df.drop(["Sales", "Newspaper"], axis=1)
y = df["Sales"]

X_train, X_test, y_train, y_test = \
  train_test_split(X, y, test_size=0.20, random_state=42)


# In[ ]:


# Se instancia el modelo de regresión lineal
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)


# In[ ]:


# Ecuación de regresión y métricas
print(f"Regresión: y = {lm.intercept_:.4f} + {lm.coef_[0]:.4f} * TV + {lm.coef_[1]:.4f} * Radio")
print(f"R2: {r2_score(y_test, y_pred):.4f}")
print(f"Correlación: {pearsonr(y_test, y_pred)[0]:.4f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred) * 100:.4f} %")


# In[ ]:





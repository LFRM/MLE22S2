#!/usr/bin/env python
# coding: utf-8

# # PD7

# Previamente en el curso hemos explorado métodos de regresión lineal para tareas predictivas. Además, hemos tratado de mejorar el poder predictivo de nuestros modelos utilizando métodos como Componentes Principales, Ridge, o Lasso. Sin embargo, todos estos asumen que nuestra relación entre Y y X es lineal, por lo que necesitamos una forma de alejarnos de este supuesto y ser capaces de modelar procesos que no cumplan con esta concición.
# 
# <center><img src="https://s3-eu-west-1.amazonaws.com/cjp-rbi-estatesgazette/wp-content/uploads/sites/3/2021/04/iStock-697940466-1-1024x682.jpg" alt="drawing" width="550"/></center>
# 
# Se le proporciona la base de datos Wage, que contiene información sobre el ingreso de hombres en la región atlántico-central de EE.UU entre 2003 y 2009. 
# 
# |Columnas|Tipo|Descripción|
# |---|---|---|
# |year|Continua |Año en que se registró la información salarial|
# |age|Continua |Edad del trabajador|
# |maritl|Categórica|Un factor con niveles 1. Nunca Casado 2. Casado 3. Viudo 4. Divorciado y 5. Separado indicando estado civil|
# |race|Categórica |Un factor con niveles 1. Blanco 2. Afrodescendiente 3. Asiático y 4. Otro indicador de raza|
# |education|Categórica |Un factor con niveles 1. < Graduado de preparatoria 2. Graduado de preparatoria 3. Algo de universidad 4. Graduado de universidad y 5. Título avanzado que indica el nivel de educación|
# |region|Categórica |Región del país (solo en el Atlántico medio.|
# |jobclass|Categórica |Un factor con niveles 1. Industrial y 2. Información que indica el tipo de trabajo|
# |health|Categórica |Un factor con niveles 1. <=Bueno y 2. >=Muy bueno que indica el nivel de salud del trabajador|
# |health_ins|Categórica |Un factor con niveles 1. Si y 2. No indicando si el trabajador tiene seguro de salud|
# |logwage|Continua|Registro de salario de los trabajadores|
# |wage|Continua |Salario bruto de los trabajadores|
# 
# Se le pide que desarrolle los siguientes pasos:
# 
# 1. Cargue los datos y realice el preprocesamiento
# 2. Realice un análisis de regresión simple para analizar la relación entre el salario y la edad.
# 3. Realice una regresión polinómica del salario sobre la edad de la persona.
# 4. Realice la estimación con Nadaraya-Watson con un kernel gaussiano y un $h=5$
# 5. Compare el desempeño de los modelos a través del MAE train y test

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from statsmodels.nonparametric.kernel_regression import KernelReg


# ## 1. Lectura de datos y preprocesamiento

# In[ ]:


# Se descarga el dataset
get_ipython().system('curl -L -O https://github.com/JWarmenhoven/ISLR-python/raw/master/Notebooks/Data/Wage.csv')


# In[ ]:


# Se lee el dataset
df_wage = pd.read_csv("Wage.csv")


# In[ ]:


# Se muestran algunos registros del dataset
df_wage.sample(5)


# In[ ]:


# Se identifican las columnas que no varian
print(df_wage["sex"].value_counts())
print()
print(df_wage["region"].value_counts())


# In[ ]:


# Se eliminan columnas innecesarias
df_wage.drop(labels=["Unnamed: 0", "sex", "region"], axis=1, inplace=True)


# In[ ]:


# Se muestra la dimension del dataset
df_wage.shape


# ## 2. Regresión simple

# In[ ]:


# Separacion de los datos en conjunto de entrenamiento y de prueba
X = df_wage[["age"]].copy()
y = df_wage["wage"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Se ajusta un modeo lineal con los 3 pasos de scikit learn: instanciar modelo, fit y predict
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)


# In[ ]:


# Guardamos la metrica para una regresion lineal simple
lm_mae = mean_absolute_error(y_test, y_pred)
print("MAE:", lm_mae)


# In[ ]:


# Ploteamos la regresion lineal simple
lm_eq =  lm.intercept_ + lm.coef_ * range(X_test.min()[0], X_test.max()[0])

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color="black", alpha=0.5)
plt.plot(range(X_test.min()[0], X_test.max()[0]), lm_eq, color="red")
plt.show()


# ## 3. Regresión polinómica

# Sin embargo, vemos que existen ciertos problemas al asumir la linealidad de los datos, por lo que probablemente ganemos poder predictivo si asumimos una forma de la estimación del tipo:
# $$y_i=\beta_0+\beta_1x_i+\beta_2x_i^2+\beta_3x_i^3+...+\beta_dx_i^d+\epsilon_i$$

# In[ ]:


# Crear variables con diferentes potencias de la variable age y generamos los conjuntos de datos de entrenamiento y prueba
X["age_2"] = X["age"] ** 2
X["age_3"] = X["age"] ** 3
X["age_4"] = X["age"] ** 4
X["age_5"] = X["age"] ** 5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Se ajusta un modeo lineal con los 3 pasos de scikit learn: instanciar modelo, fit y predict
lm = LinearRegression()
lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)


# In[ ]:


# Guardamos la metrica para una regresion lineal simple
pol_mae = mean_absolute_error(y_test, y_pred)
print("MAE:", pol_mae)


# In[ ]:


# Generamos la matriz de disenho y la ecuacion de regresion
design_mat = pd.DataFrame(range(X_test.min()[0], X_test.max()[0]), columns=["x"])
design_mat["x_2"] = design_mat["x"] ** 2
design_mat["x_3"] = design_mat["x"] ** 3
design_mat["x_4"] = design_mat["x"] ** 4
design_mat["x_5"] = design_mat["x"] ** 5

pol_eq = lm.intercept_ + np.matmul(design_mat, lm.coef_)

# Ploteamos la regresion polinomica
plt.figure(figsize=(10, 6))
plt.scatter(X_test["age"], y_test, color="black", alpha=0.5)
plt.plot(range(X_test.min()[0], X_test.max()[0]), pol_eq, color="red")
plt.show()


# ## 4. Regresión Kernel

# In[ ]:


# Separacion de los datos en conjunto de entrenamiento y de prueba
X = df_wage[["age"]].copy()
y = df_wage["wage"].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


# Se ajusta el modelo (KernelReg proviene de otra libreria distinta a scikit learn)
ndw = KernelReg(y_train, X_train, "c", bw=[5])
y_pred = ndw.fit(X_test)[0]


# In[ ]:


# Guardamos la metrica para una regresion lineal simple
ndw_mae = mean_absolute_error(y_test, y_pred)
print("MAE:", ndw_mae)


# In[ ]:


# Ploteamos la regresion kernel
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color="black", alpha=0.5)
plt.plot(X_test.sort_values(by="age")["age"], ndw.fit(X_test.sort_values(by="age"))[0], c="red")
plt.show()


# ## 5. Métricas de comparación

# In[ ]:


# Se genera un dataframe con los MAEs de los modelos para compararlos mejor
metrics = pd.DataFrame({"Regresion Lineal": [lm_mae], "Regresion Polinomica": [pol_mae], "Regresion Kernel": [nwd_mae]})
metrics.index = ["MAE"]
metrics


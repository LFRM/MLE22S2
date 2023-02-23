#!/usr/bin/env python
# coding: utf-8

# # Práctica Dirigida 2 - Regresión Lineal

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import summary_table
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr


# Considere la base de datos Advertising. Esta contiene datos de las unidades vendidas de un producto en 200 tiendas diferentes, así como del gasto en publicidad por televisión, radio y periódico para cada tiendas. 
# 
# |Columnas|Tipo|Descripción|
# |---|---|---|
# |TV|float|Presupuesto de publicidad del producto para televisión.|
# |Radio|float|Presupuesto de publicidad del producto para radio.|
# |Newspaper|float|Presupuesto de publicidad del producto para periódicos.|
# |Sales|float|Unidades vendidas del producto.|
# 
# En base a estos datos responda a las siguientes preguntas:
# 
# 1)	Realice el análisis exploratorio de los datos. \\
# > a)	Elimine columnas innecesarias. \\
#   b)	Examine la distribución de las variables. \\
#   c)	Resuelva las observaciones nulas y outliers en caso sea necesario. \\
#   d)	Analice gráficamente la distribución de los datos a través de box plots y 
#   scatter plots.
# 

# In[ ]:


# Se descarga el dataset
get_ipython().system('curl -L -O https://github.com/JWarmenhoven/ISLR-python/raw/master/Notebooks/Data/Advertising.csv')
df_advert= pd.read_csv("Advertising.csv")


# ## Exploración de los datos

# In[ ]:


df_advert.shape


# In[ ]:


df_advert.head()


# In[ ]:


df_advert.drop("Unnamed: 0",axis=1,inplace=True)


# In[ ]:


df_advert.head()


# In[ ]:


df_advert.describe()


# In[ ]:


# Detección de Outliers
fig,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(10,5))
ax1.boxplot([df_advert['TV'],df_advert['Radio'],df_advert['Newspaper']]);
ax2.boxplot(df_advert['Sales']);
plt.show()


# Los gráficos de caja nos muestran que los gastos en publicidad por radio y periódico son proporcionalmente menores al gasto en televisión. Por otro lado, podemos notar que hay dos observaciones outliers en la distribución de publicidad por periódico. Es decir que presentan un valor inusualmente alto dados los cuartiles de la distribución.

# In[ ]:


# Matriz de correlaciones
df_advert.corr()


# In[ ]:


fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
ax1.scatter(df_advert["TV"],df_advert["Sales"],c="blue")
ax2.scatter(df_advert["Radio"],df_advert["Sales"],c="red")
ax3.scatter(df_advert["Newspaper"],df_advert["Sales"],c="green")
ax1.set_xlabel("TV")
ax2.set_xlabel("Radio")
ax3.set_xlabel("Newspaper")
ax1.set_ylabel("Sales")
plt.show()


# ## Regresión Simple

# 2)	Realice un análisis de regresión simple. \\
# >a)	Estime la regresión univariada de ventas en términos de uno de los gastos en publicidad. \\
# b)	Interprete los coeficientes de la regresión. \\
# c)	Interprete la significancia de los coeficientes, así como sus intervalos de confianza. \\
# d)	Interprete los estadísticos de bondad de ajuste de la regresión. \\
# e)	Calcule la distancia de Cook para cada observación. Comente. \\
# f)	Calcule el factor de inflación de varianza para cada variable. Comente \\
# 

# In[ ]:


# Modelo de regresión simple respecto al gasto en publicidad televisiva
lm = smf.ols('Sales ~ TV',data = df_advert).fit()
print(lm.summary())


# Interpretación:
# 
# - Cuando el gasto en publicidad televisiva es cero, las ventas esperadas equivalen a 7.03 mil unidades. Luego, por cada sol adicional invertido en publicidad televisiva, las ventas esperadas aumentan en 0.05 mil unidades.
# - El coeficiente asociado al gasto en publicidad b1 tiene un t-calculado = 17.4 y un p_valor cercano a cero. Ello implica que la probabilidad de observar un t_cal de 17.68 asumiendo que b1=0 es cercana a cero, por lo cual tenemos fuerte evidencia estadística para afirmar que b1 es distinto de cero.
# - El intervalo de confianza para el coeficiente b1 es (0.042 , 0.053). Podemos deducir que este intervalo tiene un 95% de probabilidad de contener al verdadero valor del parámetro. Ello confirma la conclusión de que el parámetro diferente de cero.
# - El criterio de información de Akaike (AIC) nos muestra un valor de 1033, lo cual no nos brinda información en sí mismo, sino que debe ser comparado con otros modelos para elegir el que otorgue mayor ajuste y sea más parsimonioso.
# - La regresión muestra un R cuadrado de 61% y un R cuadrado ajustado similar. La razón por que ambos son parecidos en la regresión univariada es que el R cuadrado ajustado penaliza la métrica entre mayor es el número de features utilizados en la regresión. Podemos de ducir que el 61% de la variabilidad en las ventas es explicada por el gasto en publicidad televisiva.

# In[ ]:


st,data,ss2= summary_table(lm, alpha=0.05)
fittedvalues = data[:, 2]
predict_mean_ci_low, predict_mean_ci_upp,predict_ci_low, predict_ci_upp = data[:, 4:8].T

df_advert['predict_mean_ci_low']= pd.DataFrame(predict_mean_ci_low)
df_advert['predict_mean_ci_upp']= pd.DataFrame(predict_mean_ci_upp)
df_advert['predict_ci_low']= pd.DataFrame(predict_ci_low)
df_advert['predict_ci_upp']= pd.DataFrame(predict_ci_upp)
df_advert['fittedvalues']= pd.DataFrame(fittedvalues)
df_advert["std_errors"] = (df_advert["Sales"]-df_advert['fittedvalues'])/((df_advert["Sales"]-df_advert['fittedvalues']).std())
df_graf = df_advert.sort_values('TV')

fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
ax1.scatter(df_graf["TV"],df_graf["Sales"],c="blue")
ax1.plot(df_graf["TV"], df_graf['fittedvalues'], c='orange')
ax1.plot(df_graf["TV"],df_graf['predict_mean_ci_low'], c='red',lw=2)
ax1.plot(df_graf["TV"], df_graf['predict_mean_ci_upp'], c='red',lw=2)
ax1.plot(df_graf["TV"],df_graf['predict_ci_low'], c='green',lw=2)
ax1.plot(df_graf["TV"], df_graf['predict_ci_upp'], c='green',lw=2)
ax1.set_xlabel("Gasto en TV")
ax1.set_ylabel("Valores predichos")

ax2.scatter(df_graf["TV"],df_graf["std_errors"],c="blue")
ax2.plot(df_graf["TV"],df_graf["TV"]*0,c='orange')
ax2.set_xlabel("Gasto en TV")
ax2.set_ylabel("Errores")
plt.show()


# De los gráficos podemos deducir:
# - Se puede observar que los errores de la regresión están centrados en cero por construcción. Sin embargo, la varianza del error no es constante en todo el rango de la variable TV, lo que implica heterosedasticidad.
# - El intervalo de predicción predice en qué rango caerá una observación individual futura con un grado de confianza (e.g 95%), mientras que un intervalo de confianza muestra el rango probable de valores asociados con algún parámetro estadístico de los datos, como la media condicional (eg. 95%).

# In[ ]:


# Apalancamiento de las observacinones
print('average leverage: ', 2/(df_graf.shape[0]))
fig = sm.graphics.influence_plot(lm)


# - La medida de apalancamiento está entre 1/n y 1. El apalancamiento promedio de las observaciones debería ser (p+1)/n, donde p: número de regresores y n: número de observaciones. En este caso, hay varias observaciones con un apalancamiento de hasta 0.02.
# - Por otro lado, los errores estandarizados suelen estar entre -3 y 3 errores estándar. Por lo que ninguna de las observaciones parece estar excesivamente alejada de cero.

# In[ ]:


# ¿Cómo replicaría este análisis para el gasto en publicidad por radio y por periódico?
# Tenga en cuenta la eliminación de outliers en caso sea necesario.


# ## Regresión Multivariada

# 3)	Realice un análisis de regresión múltiple. \\
# >a)	Estime la regresión de ventas en términos de los tres tipos de gasto. \\
# b)	Repita los pasos del punto 2 de los acápites b) al d). \\
# c)	Si alguna variable debe ser eliminada por ser poco significativa, hágalo y luego repita el acápite f) del punto 2). \\
# d)	Si algunas observaciones deben ser eliminadas por ser outliers, hágalo y luego realice un análisis predictivo a través de regresión lineal y genere estadísticos de desempeño para la predicción del modelo en la data de prueba.
# 

# In[ ]:


# # Modelo de regresión simple respecto al gasto en publicidad por televisión, radio y periódico
lm = smf.ols('Sales ~ TV + Radio + Newspaper',data = df_advert).fit()
print(lm.summary())


# Interpretación:
# - El intercepto de la regresión nos muestra un valor hipotético del nivel de ventas en el caso que el gasto en los tres tipos de publicidad sea cero. En este caso sería de 2.93 mil unidades. Sin embargo, podemos estar sujetos a error de extrapolación.
# - El gasto en publicidad por radio tienen un efecto positivo y estadísticamente significativo en las ventas (b1=0.05, b2=0.19 y p1=p2=0). Mientras que el gasto en publicidad por periódico tiene un efecto negativo y poco significativo en las ventas (b3=-0.001 y p3=0.86).
# - Notemos que la variable 'Newspaper' tiene relativamente baja correlación con 'Sales' y relativamente alta correlación con 'Radio', lo que podría explicar en parte la poca significancia que tiene en el modelo. Idealmente, nos gustaría encontrar lo opuesto en nuestros features.
# - Los intervalos de confianza muestran que los verdaderos valores de b1, b2 y b3 estarían con un 95% de confianza entre los intervalos (0.043, 0.049),(0.172, 0.206) y (-0.013, 0.011).
# - El R cuadrado aumentó de 61.2% a 89.7%. El R2 ajustado no muestra pérdida de poder explicativo debido a excesivo número de predictores. 
# - El criterio de información de Akaike fue de 780.4, una mejora con respecto al 1042 del modelo univariado. Lo que muestra que el modelo multivariado tiene mayor bondad de ajuste que el modelo univariado, mientras que sigue siendo parsimonioso.
# 

# In[ ]:


# # Modelo de regresión simple respecto al gasto en publicidad por televisión y radio.
lm = smf.ols('Sales ~ TV + Radio',data = df_advert).fit()
print(lm.summary())


# In[ ]:


st,data,ss2= summary_table(lm, alpha=0.05)
fittedvalues = data[:, 2]

df_advert['fittedvalues']= pd.DataFrame(fittedvalues)
df_graf = df_advert.sort_values('Sales')
df_graf["std_errors"] = (df_graf["fittedvalues"]-df_graf['Sales'])/((df_graf["fittedvalues"]-df_graf['Sales']).std())

fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
ax1.scatter(df_graf["Sales"],df_graf["fittedvalues"],c="blue")
ax1.plot(df_graf["Sales"], df_graf['Sales'], c='orange')
ax1.set_xlabel("Valores reales")
ax1.set_ylabel("Valores predichos")

ax2.scatter(df_graf["Sales"],df_graf["std_errors"],c="blue")
ax2.plot(df_graf["Sales"],df_graf["Sales"]*0,c="orange")
ax2.set_xlabel("Valores reales")
ax2.set_ylabel("Errores")
plt.show()


# - En un modelo multivariado no se visualiza X vs fitted_values, sino values vs fitted_values. 
# - Para visualizar los intervalos de confianza y de predicción necesitaríamos graficar un espacio vectorial de 4 dimiensiones, lo cual no es factible, por lo que se descarta el uso de estos.
# - Se puede observar que hay una alta correlación entre los valores predichos y realizados de las ventas. La mayoría de puntos en el gráfico de regresión se ubican a lo largo de la recta de 45°, lo que demuestra la alta bondad de ajuste del modelo.
# - Al inspeccionar el gráfico de errores podemos observar que la varianza del error aún no es constante, pero la heterosedasticidad es menor que en el caso del modelo univariado.

# In[ ]:


# Apalancamiento de las observacinones
print('average leverage: ', 3/(df_graf.shape[0]))
fig = sm.graphics.influence_plot(lm)


# - En este caso, 1 observación tienen un valor de apalancamiento que es mayor al doble del valor promedio esperado (0.015). Esto indicaría que son observaciones inusuales.
# - Por otro lado, dos observaciones tienen un error estándar menor a -3 veces el error estándar, lo cual podría sesgar la estimación de la regresión. Se recomienda eliminar estas observaciones para evitar sesgos en los parámetros.

# In[ ]:


# # Modelo de regresión simple respecto al gasto en publicidad por televisión y radio eliminando outliers
df_advert.drop([5,130],axis=0,inplace=True)
lm = smf.ols('Sales ~ TV + Radio',data = df_advert).fit()
print(lm.summary())


# Después de la eliminación de las observaciones outlier obseramos una mejora en los estadísticos de la regresión. el R cuadrado aumentó de 0.897 a 0.915, mientras el AIC se redujo de 778.4 a 728.3, lo que indica un modelo con mayor bondad de ajuste.

# In[ ]:


# Análisis predictivo utilizando muestras de train y test
X = df_advert[['TV','Radio']]
y = df_advert['Sales']
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42)
lm = LinearRegression().fit(X_train,y_train)
y_pred = lm.predict(X_test)

print('El R2 del test es de: ',lm.score(X_test,y_test))
print('El MAE del test es de: ', mean_absolute_error(y_test,y_pred))
print('El MAE naive del test es de: ', ((y_test-(y_test.mean())).abs()).mean())
print('La correlación entre el la predicción y el real es: ', pearsonr(y_pred,y_test)[0])

plt.scatter(y_test,y_pred,c='blue')
plt.plot(y_test,y_test,c='orange')
plt.xlabel('valores reales')
plt.ylabel('valores predichos');


# # Fórmulas:
# 
# MAE:\
# $$MAE=\sum_{i=1}^n \frac{|y_i-\hat{y}_i|}{n}, \ \ MAE_{Naive}=\sum_{i=1}^n \frac{|y_i-\bar{y}_i|}{n}$$
# R cuadrado univariado: \
# $$R^2=\frac{\sum^n_{i=1}\hat{y}_i-\bar{y}}{\sum^n_{i=1}y_i-\bar{y}}=1-\frac{RSS}{TSS}$$
# 
# Rcuadrado ajustado: \
# $$Ad R^2=1-\frac{(1-R^2)(n-1)}{n-p-1}$$
# 
# T-calc:\
# $$T_{calc}=\frac{b}{SE_b} , SE_b=\sqrt{\frac{\frac{1}{n-2}\sum_{i=1}^n(y_i-\hat{y}_i)^2}{\sum_{i=1}^n(x_i-\bar{x}_i)^2}}$$
# 
# AIC:\
# $$AIC=2(k-ln(L))$$
# <center> donde $k$ es el número de parámetros estimados y $L$ es el máximo valor de la función de verosimilitud para el modelo. </center>
# 
# Confidence Interval:
# $$CI = \left( \hat{\beta} - Z_{\frac{\alpha}{2}}\frac{S}{\sqrt{n}},\hat{\beta} + Z_{\frac{\alpha}{2}}\frac{S}{\sqrt{n}} \right)$$
# 
# H leverage:
# 
# $$h_{ii}=\frac{\partial\hat{y}_i}{\partial y_i}=[H]_{ii}=x_i(X^{\top}X)^{-1}x_i^{\top}$$
# 
# El estadístico de apalancamiento de una observación $i$ es el elemento $ii$ de la matriz de orto-proyección de los datos de $y$ sobre el espacio vectorial de las $X$. Esto nos da una noción de qué tan alejada está la observación del valor esperado por el modelo. 

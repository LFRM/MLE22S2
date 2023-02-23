#!/usr/bin/env python
# coding: utf-8

# Universidad del Pacífico \
# Departamento Académico de Economía \
# Machine Learning para Economistas \
# Segundo Semestre de 2022 \
# Profesor: F. Rosales, JP: R. Aráuco
# 
# 
# # Práctica Dirigida 3 - Selección de Modelos de Clasificación 
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import mlxtend as sfs

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, make_scorer


# La base Default contiene datos de usuarios de un banco, de los cuales algunos incumplieron en su deuda de tarjeta de crédito a lo largo del año. Además, se tienen datos sobre el ingreso anual de los deudores, su balance promedio de deuda mensual, y una variable binaria que indica si son estudiantes o no. Se quiere desarrollar un modelo que prediga acertadamente si un individuo incumplirá en su deuda en base a las características previamente mencionadas. Para ello, se le pide seguir los siguientes pasos:
# 
# 1. Cargue la base de datos y realice el preprocesamiento.
# 2. Realice el análisis exploratorio de datos.
# 3. Corra una regresión logística de la variable 'default' contra los predictores disponibles. Halle la matriz de confusión y métricas de desempeño. Describa la diferencia entre las métricas de Accuracy, Precision, Recall, F1 Score, Sensitivity y Specificity. Luego, grafique la curva ROC, determine el área bajo la curva e interprete.
# 4. Estime un modelo KNN determinando el óptimo número de K por cross-validation. 
# 5. Halle la matriz de confusión y las métricas de desempeño.
# 6. Compare los resultados de la predicción por KNN con la regresión logística.

# ## 1. Preprocesamiento

# In[ ]:


get_ipython().system('curl -L -O https://raw.githubusercontent.com/JWarmenhoven/ISLR-python/master/Notebooks/Data/Hitters.csv')
get_ipython().system('curl -L -O https://github.com/JWarmenhoven/ISLR-python/raw/master/Notebooks/Data/Default.xlsx')


# In[ ]:


df_default = pd.read_excel('Default.xlsx')


# In[ ]:


df_default.shape


# In[ ]:


df_default.head()


# In[ ]:


df_default.dtypes


# In[ ]:


df_default['default'].replace(('Yes','No'),(1,0),inplace=True)
df_default['student'].replace(('Yes','No'),(1,0),inplace=True)


# In[ ]:


df_default.dtypes


# ## 2. Análisis exploratorio

# In[ ]:


df_default.head()


# In[ ]:


df_default.describe()


# Nótese que solo el 3% de los deudores efectivamente incumplieron en su deuda. Lo que implica que el dataset está desbalanceado con respecto a las personas que hicieron vs que no incumplieron.

# In[ ]:


graf = pd.concat([df_default.loc[df_default['default']==0].sample(n=300),df_default.loc[df_default['default']==1]],axis=0)
fig, (ax1, ax2) = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
ax1.scatter(graf[graf['default']==1]['balance'],graf[graf['default']==1]['income'],c='red',label='Default')
ax1.scatter(graf[graf['default']==0]['balance'],graf[graf['default']==0]['income'],c='blue',label='Non default')
ax2.scatter(graf[graf['student']==1]['balance'],graf[graf['student']==1]['income'],c='yellow',label='Student')
ax2.scatter(graf[graf['student']==0]['balance'],graf[graf['student']==0]['income'],c='green',label='Non student')
ax1.set_xlabel('Balance')
ax2.set_xlabel('Balance')
ax1.set_ylabel('Income')
fig.legend(loc='upper center', bbox_to_anchor=(0.2, 0),ncol=2);


# - Ambos gráficos muestran la distribución conjunta de balance de deuda e ingreso anual. En el gráfico de la izquierda el amarillo muestra aquellos que hicieron default en su deuda, en morado están algunos de los que no lo hicieron. En el gráfico de la derecha el amarillo muestra aquellos deudores que son estudiantes, en morado están los que no lo son.
# - Podemos concluir que los individuos con mayores balances de deuda promedio mensuales tienden a ser aquellos que hacen default con mayor probabilidad. Mientras tanto, los individuos con menor ingreso anual suelen ser aquellos que son estudiantes.
# - Finalmente, el nivel de ingreso no es una variable que ayude en gran medida a diferenciar a los deudores morosos. El estatus de estudiante tampoco está relacionado en gran medida con la probabilidad de defauld de un deudor.

# ## 3. Regresión Logística

# In[ ]:


X = df_default[['balance','income','student']]
y = df_default['default']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
scale = StandardScaler()
X_train_log = scale.fit_transform(X_train)
X_test_log = scale.transform(X_test)


# In[ ]:


# Modelo de predicción de default en base a balance de deuda
lr = LogisticRegression()
lr.fit(X_train_log,y_train)
y_pred = lr.predict(X_test_log)
print('Matriz de confusión Reg Logística:')
pd.DataFrame(confusion_matrix(y_test, y_pred),columns=['Pred 0','Pred 1'],index=['Real 0','Real 1'])


# - ¿Por qué no utilizamos Regresión Lineal?: Regresión lineal no es el método más apropiado para clasificación porque no garantiza que el outcome esté entre 0 y 1 (que es la probabilidad estimada).
# - ¿Qué métrica es mejor para evaluar nuestro modelo?: Primero tenemos que ver la estructura de nuestro set de datos y también nuestro objetivo. El objetivo principal es detectar deudores morosos. Pero, ¿qué es más costoso?, ¿confundir un cliente moroso por cumplido o viceversa? Estas preguntas nos ayudarán a saber qué métrica nos ayudará a obtener el mejor modelo.
# - Al final de la PD encontrarán las fórmulas de las diferentes métricas de performance.

# - **Accuracy**: Esta métrica nos indica que proporción de las observaciones hemos podido clasificar correctamente en su categoría correspondiente. Debido a que la data es muy desbalanceada, si predecimos todo con 0 (no fraudulento), nuestro accuracy será bastante alto.
# - **Precision**: Esta métrica nos indica cuál es la proporción de las observaciones que predecimos como morosos, de verdad lo son. Maximizar esta métrica implicaría minimizar falsos positivos, es decir, buscar por todos los medios no considerar deudores cumplidos como morosos. Un modelo con buena precisión, cada vez que predice un deudor moroso, tiene una probabilidad muy alta de que de verdad lo sea.
# - **Recall**: Esta métrica nos indica cuál es la proporción del total de deudores morosos que clasificamos como tal. Maximizar esta métrica implica minimizar los falsos negativos, es decir, hacer pasar deudores morosos como cumplidos.
# - **F1 Score**: Se calcula como la media armónica entre el recall y precision. Esta métrica permite lidiar con bases de datos desbalanceadas. No solo toma en cuenta la cantidad de errores de predicción, sino el tipo de errores que se cometen. El F1 será alto cuando sus componentes sean altos, mientras que será bajo cuando ambos componentes sean bajos.
# - **Sensitivity**: La probabilidad de que el modelo prediga un deudor moroso cuando realmente lo era. También se le llama el “true positive rate.”
# - **Specificity**: La probabilidad de que el modelo prediga un deudor cumplido cuando realmente lo era. También se le llame el 
# “true negative rate.”
# 
# Decidir cuál de estas dos es mejor, dependerá de nuestro objetivo.

# In[ ]:


accuracy_logit = accuracy_score(y_test,y_pred)
precision_logit = precision_score(y_test,y_pred)
recall_logit = recall_score(y_test,y_pred)
F1_logit = 2*((precision_score(y_test,y_pred)*recall_score(y_test,y_pred))/((precision_score(y_test,y_pred)+recall_score(y_test,y_pred))))
print('Accuracy: ', accuracy_logit)
print('Accuracy Naive: ', accuracy_score(y_test,y_test*0))
print('Precision: ', precision_logit)
print('Recall: ', recall_logit)
print('F1: ',F1_logit)


# In[ ]:


# Curva ROC
y_pred_prob = lr.predict_proba(X_test_log)[::,1]
fpr, tpr, _ = roc_curve(y_test,y_pred_prob)
plt.plot(fpr,tpr,c='blue')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC')
plt.show()
print('Área bajo la curva (AUC): ',roc_auc_score(y_test,y_pred_prob))


# - La curva ROC permite visualizar simultáneamente el true positive rate (tpr = Sensitivity) el false positive rate (fpr = 1-Specificity) de un modelo de clasificación para diferentes umbrales para la probabilidad posterior de default. 
# - Mientras más pegada está la curva a la ezquina superior izquierda mayor será la calidad del modelo. Podemos cuantificar esta cercanía midiendo el área bajo la curva, entre más cerca a 1, mejor predice el modelo. Entre más cerca a 0.5, más parecido será a la calidad de un predictor naive.

# ## 4. Predicción por KNN

# In[ ]:


knn = KNeighborsClassifier(n_neighbors=10)
X = df_default[['balance','income','student']]
y = df_default['default']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# In[ ]:


scale = MinMaxScaler()
X_train_knn = scale.fit_transform(X_train)
X_test_knn =  scale.transform(X_test)


# In[ ]:


k_values=list(range(1,15))
precisions=[]
recalls=[]
for i in tqdm(range(1,15)):
    knn = KNeighborsClassifier(n_neighbors=i).fit(X_train_knn,y_train)
    scores = cross_validate(knn,X_train_knn, y_train,scoring=['recall','precision'],n_jobs=-1,cv=10)
    precisions.append(np.mean(scores['test_recall']))
    recalls.append(np.mean(scores['test_precision']))
recall_max = pd.DataFrame(recalls)
recall_max = recall_max[recall_max[0]==recall_max[0].max()].index[0] + 1
print('k óptimo: ',recall_max)


# In[ ]:


plt.plot(k_values,precisions,label='Precision')
plt.plot(k_values,recalls,label='Recall')
plt.legend();


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=recall_max).fit(X_train_knn, y_train)
y_pred_knn = knn.predict(X_test_knn)

print('Matriz de confusión KNN:')
pd.DataFrame(confusion_matrix(y_test, y_pred_knn),columns=['Pred 0','Pred 1'],index=['Real 0','Real 1'])


# ## 5. Comparación de métodos

# In[ ]:


accuracy_knn = accuracy_score(y_test,y_pred_knn)
precision_knn = precision_score(y_test,y_pred_knn)
recall_knn = recall_score(y_test,y_pred_knn)
F1_knn = 2*((precision_score(y_test,y_pred_knn)*recall_score(y_test,y_pred_knn))/((precision_score(y_test,y_pred_knn)+recall_score(y_test,y_pred_knn))))

comp =[[accuracy_logit,accuracy_knn],[precision_logit,precision_knn],[recall_logit,recall_knn],[F1_logit,F1_knn]]
comp = np.round(comp,4)
pd.DataFrame(comp,index=['Accuracy','Precision','Recall','F1'],columns=['Logit','KNN'])


# En comparación a la regresión logística, el método de KNN muestra un peor desempeño ya que es capaz de identificar correctamente a 33 de los 113 deudores que realmente incumplieron, mientras que la regresión logística idetificó a 39 de estos. Esto se ve reflejado en un mejor score recall de 29% para la KNN vs un 35% para regresión logística.

# ## Feature selection - Backward & Forward selection

# Por otro lado, se le proporciona la base de datos Hitters. Esta contiene datos de 322 jugadores de las Grandes Ligas de Béisbol en las temporadas de 1986 y 1987. La base muestra diferentes atributos de su desempeño y experiencia profesional, así como su salario anual en el comienzo de la temporada en miles de dólares. Se le pide crear un modelo que prediga la probabilidad de que un beisbolista pertenezca al cuartil más alto de la distribución de salario anual teniendo en cuenta sus atributos y desempeño. Para ello, haga lo siguiente:

# 1. Cargue los datos y elabore el preprocesamiento de la base
# 2. Realice el análisis exploratorio
# 3. Cree una variable binaria que identifique si un jugador pertenece al cuarto cuartil de la distribución.
# 4. Estime un modelo de regresión logística para predecir si un jugador pertenecerá o no al cuartil más alto de la distribución. Calcule la métrica Precision. 
# 5. Elija los features de la regresión usando forward selection y como métrica de referencia el Precision. 
# 6. Construya la matriz de confusión del modelo y calcule la métrica Precision del modelo final. Interprete
# 7. Compare sus resultados en la pregunta 6. con los resultados en la pregunta 4.

# ## 1. Preprocesamiento

# In[ ]:


df_salary = pd.read_csv('Hitters.csv')
df_hitters = df_salary.copy()
df_hitters.shape


# In[ ]:


df_hitters.head()


# In[ ]:


for i in df_hitters.columns: 
    if df_hitters[i].isnull().sum()>0: print([i, df_hitters[i].isnull().sum()])

df_hitters.dropna(axis=0,inplace=True)
df_hitters.drop('Unnamed: 0',axis=1,inplace=True)
df_hitters.reset_index(inplace=True,drop=True)
df_salary.shape, df_hitters.shape


# In[ ]:


df_hitters = pd.get_dummies(df_hitters,columns=['League','Division','NewLeague'],prefix=['League','Division','NL'],drop_first=True)
df_hitters.head()


# ## 2. Análisis exploratorio

# In[ ]:


df_hitters.describe()


# In[ ]:


corrmat = round(df_hitters.corr(),2)
mask = np.triu(np.ones_like(corrmat, dtype=bool))
f, ax = plt.subplots(figsize=(15,15))
sns.heatmap(corrmat, vmax=.8, square=True,annot=True,mask=mask)


# ## 3. División por cuartiles

# In[ ]:


print('Cuartiles de la distribución: ', np.quantile(df_hitters['Salary'],[.23,.5,.75]))
df_hitters['top_salary']=(df_hitters['Salary']>=np.quantile(df_hitters['Salary'],0.75))*1
pd.DataFrame(df_hitters['top_salary'].value_counts())


#  ## 4. Regresión Logística

# In[ ]:


# Hacemos la separación de objetivo y predictores, el train-test split y estandarizamos
target = 'top_salary'
X = df_hitters.drop(['Salary',target],axis=1)
y = df_hitters[target]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

scale = StandardScaler()
X_train_log = pd.DataFrame(scale.fit_transform(X_train),columns =X.columns)
X_test_log = pd.DataFrame(scale.transform(X_test),columns=X.columns)


# In[ ]:


#Corremos un modelo de regresión logística con todas las variables como base de comparación
lr = LogisticRegression()
lr.fit(X_train_log,y_train)
y_pred = lr.predict(X_test_log)

precision1 = precision_score(y_test,y_pred)
print('Matriz de confusión Reg logística:')
pd.DataFrame(confusion_matrix(y_test, y_pred),columns=['Pred 0','Pred 1'],index=['Real 0','Real 1'])


# In[ ]:


# Definimos la métrica de scoring como Precision
def prec(y_true,y_pred):
    return precision_score(y_true,y_pred,zero_division=0)
scoring = make_scorer(prec)

# Realizamos selección de predictores para quedarnos con aquellos que ayudan a maximizar el Precision
lr = LogisticRegression()
sfs = SequentialFeatureSelector(lr,cv=5,direction='forward',scoring=scoring)
sfs.fit(X_train_log,y_train)
X_train_fs = sfs.transform(X_train_log)
X_test_fs = sfs.transform(X_test_log)

print('Número inicial de variables: ',X_train.shape[1])
print('Número de variables seleccionadas: ',X_train_fs.shape[1])
print('Variables seleccionadas: ', sfs.get_feature_names_out())


# In[ ]:


# Crorremos la regresión logística y hallamos la matriz de confusión
lr = LogisticRegression()
lr.fit(X_train_fs,y_train)
y_pred = lr.predict(X_test_fs)

print('Matriz de Confusión Reg Logística:')
pd.DataFrame(confusion_matrix(y_test, y_pred),columns=['Pred 0','Pred 1'],index=['Real 0','Real 1'])


# In[ ]:


# Calculamos las métricas de desempeño y comparamos
precision2 = precision_score(y_test,y_pred)

comp = [[precision1,precision2]]
comp = np.round(comp,4)
comp = pd.DataFrame(comp,index=['Precision'],columns=['Sin FS','Con FS'])
comp


# Se puede observar que el método de forward selection nos permitió formular un modelo con solo 9 predictores y que tiene un performance ligeramente mejor que el modelo original con 19 predictores. El modelo con FS identificó acertadamente a 13 de los 16 jugadores que clasificó como pertenecientes al cuartil más alto de ingresos. Mientras tanto, el modelo original solo acertó en 12 de los 16 jugadores que predijo como pertenecientes al cuartil más alto. Esto se ve reflejado en que el modelo con FS tiene un precision de 81% mientras el original tiene solo 75%.

# # Fórmulas:
# 
# ||Pred. Positive|Pred. Negative|
# |-|----------|---------------------|
# |Actual Positive|True Positive (TP)|False Positive (FP)|
# |Actual Negative|False Negative (FN)|True Negative (TN)|
# 
# - Accuracy: $$\frac{TP+TN}{TP+TN+FP+FN}$$
# 
# - Precision: $$\frac{TP}{TP+FP}$$
# 
# - Recall: $$\frac{TP}{TP+FN}$$
# 
# - F1: $$2 \left( \frac{precision * recall}{precision+recall} \right) $$
# 
# 

# In[ ]:





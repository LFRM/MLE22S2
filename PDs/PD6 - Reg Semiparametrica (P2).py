#!/usr/bin/env python
# coding: utf-8

# Universidad del Pacífico \
# Departamento Académico de Economía \
# Machine Learning para Economistas \
# Segundo Semestre de 2022 \
# Profesor: F. Rosales, JP: R. Aráuco
# 
# 
# # Práctica Dirigida 8 - Métodos de regresión semiparamétrica

# In[1]:


get_ipython().system('pip install pygam')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, confusion_matrix, accuracy_score, f1_score, roc_auc_score, roc_curve
from pygam import LinearGAM, LogisticGAM
from statsmodels.nonparametric.kernel_regression import KernelReg
from scipy.interpolate import UnivariateSpline


# Previamente en el curso hemos explorado métodos de regresión lineal para tareas predictivas. Además, hemos tratado de mejorar el poder predictivo de nuestros modelos utilizando métodos como Componentes Principales, Ridge, o Lasso. Sin embargo, todos estos asumen que nuestra relación entre Y y X es lineal, por lo que necesitamos una forma de alejarnos de este supuesto y ser capaces de modelar procesos que no cumplan con esta concición.

# ## Pregunta 1

# Se le proporciona la base de datos Wage, que contiene información sobre el ingreso de hombres en la región atlántico-central de EE.UU entre 2003 y 2009. Se le pide que desarrolle los siguientes pasos:
# 
# 1) Cargue los datos y realice el preprocesamiento
# 5) Realice una regresión por polinomios truncados ($q=3$)\
#     a) Realice la regresión sin penalización con diez nodos equidistantes\
#     b) Realice la regresión penalizando con un $\lambda= 3.5*10^6$
# 6) Realice una regresión por GAM lineal
# 7) Compare el desempeño de los modelos a través del MAE

# ### 1.1 Preprocesamiento

# In[3]:


# Se descarga el dataset
get_ipython().system('curl -L -O https://github.com/JWarmenhoven/ISLR-python/raw/master/Notebooks/Data/Wage.csv')


# In[4]:


df_wage = pd.read_csv('Wage.csv')
df_wage.drop(['Unnamed: 0','sex','region'],axis=1,inplace=True)
df_wage.head()


# In[5]:


df_wage.shape


# ### 1.2 Regresión lineal

# Si quisieramos estimar la relación lineal entre el salario de la persona y su edad haríamos lo siguiente

# In[6]:


# Preparación de los datos
X = df_wage[['age']].copy()
y = df_wage['wage'].copy()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

#Estimación del modelo
lm = LinearRegression()
lm.fit(X_train, y_train)
mae_lm_train = mean_absolute_error(y_train,lm.predict(X_train))
mae_lm_test = mean_absolute_error(y_test,lm.predict(X_test))
print('MAE: ',mae_lm_test)

#Gráfica
fit = np.linspace(X_test['age'].min(),X_test['age'].max(),100)
plt.scatter(X_test,y_test,c='black',alpha=0.5,linewidths=0)
plt.plot(fit,lm.intercept_+fit*lm.coef_[0],c='red');


# ### 1.5.a Polinomios truncados sin penalizar

# In[7]:


def knot_list(df,predictor,n_knots):
    knots=[]
    for i in range(int(100/(n_knots+1)),int(100/(n_knots+1))*(n_knots+1),int(100/(n_knots+1))):
        corte=np.nanpercentile(df[predictor], i)
        knots.append(corte)
    return knots


# In[8]:


knot_list(df_wage,'age',10)


# In[9]:


def generar_X_spline(df,predictor,lista_knots):
    X_return=pd.DataFrame()
    X_return['x']=df[predictor]
    X_return['x2']=df[predictor]**2
    X_return['x3']=df[predictor]**3
    for i in lista_knots:
        nombre_var='h(x,'+str(int(i))+')'
        X_return[nombre_var]=np.where(X_return['x']>i,(X_return['x']-i)**3,0)
    return X_return


# In[ ]:


# Preparación de datos
n = 10
X=generar_X_spline(df_wage,'age',knot_list(df_wage,'age',n))
y=df_wage['wage']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# Estimación del modelo
lm=LinearRegression().fit(X_train,y_train)
mae_spl_train = mean_absolute_error(y_train,lm.predict(X_train))
mae_spl_test = mean_absolute_error(y_test,lm.predict(X_test))
print('MAE: ',mae_spl_test)

# Gráfica
fit = pd.DataFrame(list(range(X_test['x'].min(),X_test['x'].max()+1)),columns=['x'])
X_aux=generar_X_spline(fit,'x',knot_list(X_test,'x',n))
plt.scatter(X_test['x'],y_test,color='black',alpha=0.5,linewidths=0)
plt.plot(list(range(X_test['x'].min(),X_test['x'].max()+1)),lm.predict(X_aux),color='red')
plt.show();


# ### 1.5.b Spline penalizado

# In[ ]:


# Preparación de los datos
X = df_wage[['age','wage']].copy()
X_train, X_test = train_test_split(X,test_size=0.3,random_state=42)
X_train , X_test = X_train.sort_values(by='age'), X_test.sort_values(by='age')

#Estimación del modelo
spline = UnivariateSpline(X_train['age'],X_train['wage'],s=3500000)
mae_ssp_train = mean_absolute_error(X_train['wage'],spline(X_train['age']))
mae_ssp_test = mean_absolute_error(X_test['wage'],spline(X_test['age']))
print('MAE:', mae_ssp_test)

# Gráfica
plt.scatter(X_test['age'],X_test['wage'],color='black',alpha=0.5,linewidths=0)
plt.plot(X_test['age'],spline(X_test['age']),color='red');


# ### 1.6 Generalized Additive Model

# In[ ]:


# Preparación de los datos
from matplotlib.lines import lineStyles

X = df_wage[['age']].copy()
y = df_wage['wage'].copy()
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

# Estimación del modelo
gm = LinearGAM()
gm.fit(X_train,y_train)
mae_gm_train = mean_absolute_error(y_train, gm.predict(X_train))
mae_gm_test = mean_absolute_error(y_test, gm.predict(X_test))
print('MAE:',mae_gm_test)

# Gráfica
conf = pd.DataFrame(gm.confidence_intervals(X_test.sort_values(by='age')))
plt.scatter(X_test['age'],y_test,c='black',alpha=0.5,linewidths=0)
plt.plot(X_test.sort_values(by='age')['age'],gm.predict(X_test.sort_values(by='age')),c='red')
plt.fill_between(X_test.sort_values(by='age')['age'],conf[0],conf[1],color='blue',alpha=0.2);


# ### 1.7 Comparación de resultados

# In[ ]:


comp = [[mae_spl_train,mae_spl_test],[mae_ssp_train,mae_ssp_test],[mae_gm_train,mae_gm_test]]
pd.DataFrame(comp,columns=['MAE train','MAE test'],index=['Reg spline truncado','Reg spline smooth','GAM'])


# Deducimos que la regresión por el estimador de Nadaraya-Watson resultó en la mejor predicción con un MAE test de 27.11. El modelo con el peor desempeño fue el de regresión lineal que fue incapaz de incorporar la no linealidad en la relación entre la edad de la persona y su salario. 

# ## Pregunta 2

# A continuación se le pide que trabaje con la base BostonMortgages, la cual contiene datos de 2380 aplicaciones a préstamos hipotecarios en Boston, EE.UU; para los años 1997-1998. La variable a predecir en esta caso es "deny" que indica si la aplicación fue rechazada (yes) o aceptada (no). En este caso, nos interesa modelar la probabilidad de que una aplicación sea rechazada. Para ello, se le pide lo siguiente:
# 
# 1) Cargue los datos y realice el preprocesamiento
# 2) Estime un modelo GAM logístico para predecir la probabilidad de que se rechace el préstamo en función a los predictores por separado. Grafique sus resultados e interprete.
# 3) Estime un modelo GAM logístico multivariado en función a las variables más relevantes del modelo. Estime una matriz de confusión del modelo multivariado, halle las métricas de desempeño y grafique la curva ROC.

# ### Glosario de variables de BostonMortgages
# 
# |Abreviación| Descripción|
# |---|---|
# |Deny| Whether application was denied|
# |dir| Debt to income ratio|
# |hir| Housing expenditure to income ratio|
# |lvr| Loan to assesed property value ratio|
# |ccs| Credit Score|
# |mcs| Mortgage Credit Score|
# |pbcr| Public bad credit rating|
# |dmi| Wether the mortgage insurance was denied|
# |self| Wether applicant is self-employed|
# |single| Whether applicant is single|
# |uria| Unemployment rate in the applicant's industry|
# |condominium| Wether the property is in a condominium|
# |black| Wether applicant is black|
# 

# ### 2.1 Preprocesamiento

# In[ ]:


# Carga de datos
df_boston = pd.read_csv('BostonMortgages.csv')
df_boston.drop('Unnamed: 0',axis=1,inplace=True)


# In[ ]:


#Transformación de datos
for i in ['deny','pbcr','dmi','self','single','black']:
    df_boston[i].replace(to_replace=['yes','no'],value=[1,0],inplace=True)
df_boston.head()
df_boston = pd.get_dummies(df_boston,columns=['ccs','mcs'],drop_first=True)


# In[ ]:


df_boston.isnull().sum(axis=0)


# In[ ]:


df_boston[['deny','black']].corr()


# In[ ]:


df_boston.describe().round(4)


# In[ ]:


for i in ['dir','hir','lvr']:
    df_boston['out'] = np.abs(df_boston[i]-df_boston[i].median())>5*np.abs(np.quantile(df_boston[i],q=.75)-np.quantile(df_boston[i],q=.25))
    df_boston.drop(df_boston[df_boston['out']==1].index,axis=0,inplace=True)

plt.boxplot(df_boston);


# ### 1.2 GAM univariado

# In[ ]:


def estimar(variable):
    #Preparación de los datos
    X = df_boston.copy()
    X_train, X_test = train_test_split(X,test_size=0.3,random_state=42)

    #Estimación del modelo
    gam = LogisticGAM()
    gam.fit(X_train[variable],X_train['deny'])
    gam.accuracy(X_test[variable],X_test['deny'])
    y_pred = gam.predict_proba(X_test[variable])

    #Gráfica
    conf = pd.DataFrame(gam.confidence_intervals(X_test.sort_values(by=variable)[variable]))
    plt.scatter(X_test[variable],X_test['deny'],c='white',edgecolors='blue',linewidths=1,alpha=0.5)
    plt.plot(X_test.sort_values(by=variable)[variable],gam.predict_proba(X_test.sort_values(by=variable)[variable]),c='black')
    plt.fill_between(X_test.sort_values(by=variable)[variable],conf[0],conf[1],color='lime',alpha=0.4)
    plt.xlabel(variable), plt.ylabel('Probability of denial')
    plt.show();

for i in ['dir','hir','lvr','pbcr','dmi','self','single','uria','condominium','black', 'ccs_2','ccs_3','ccs_4','ccs_5','ccs_6','mcs_2','mcs_3','mcs_4',]:
    estimar(i)


# ### 2.3 GAM multivariado

# In[ ]:


variable = ['dir','hir','lvr','pbcr','dmi','black','ccs_6','mcs_4',]
X = df_boston.copy()
X_train, X_test = train_test_split(X,test_size=0.3,random_state=42)

#Estimación del modelo
gam = LogisticGAM()
gam.fit(X_train[variable],X_train['deny'])
y_true = X_test['deny']
y_pred = gam.predict(X_test[variable])
y_prob = gam.predict_proba(X_test[variable])
#Métricas de desempeño
print('Accuracy:',accuracy_score(y_true,y_pred))
print('F1 Score:',f1_score(y_true,y_pred))
print('ROC AUC:', roc_auc_score(y_true,y_prob))
tp, fp,th = roc_curve(y_true,y_prob)
plt.plot(tp,fp)
plt.show();
pd.DataFrame(confusion_matrix(y_true,y_pred),index=['Predict Negative','Predict Positive'],columns=['True Negative','True Positive'])


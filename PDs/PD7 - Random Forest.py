#!/usr/bin/env python
# coding: utf-8

# # PD8

# <center><img src="https://images.ctfassets.net/9wtva4vhlgxb/3tm9zcqkxoMsJV9B2Yvg6S/ab5927edc039c2ff3741b05a2c05a63f/Best_car_seats_720x480.jpg" alt="drawing" width="550"/></center>
# 
# 
# A continuación se muestra la descripción del conjunto de datos de ventas de sillas de carro para niños en 400 tiendas distintas:

# |Columnas|Tipo|Descripción|
# |---|---|---|
# |Sales|Continua |Ventas unitarias (en miles) en cada ubicación|
# |CompPrice|Continua |Precio cobrado por el competidor en cada ubicación|
# |Income|Continua|Nivel de ingresos de la comunidad (en miles de dólares)|
# |Advertising|Continua |Presupuesto de publicidad local para la empresa en cada ubicación (en miles de dólares)|
# |Population|Continua |Tamaño de la población en la región (en miles)|
# |Price|Continua |Precio de los cargos de la compañía por los asientos de seguridad en cada sitio|
# |ShelveLoc|Categórica |Un factor con niveles Malo, Bueno y Medio que indica la calidad de la ubicación de las estanterías para los asientos de seguridad en cada sitio|
# |Age|Continua |Media de edad de la población local|
# |Education|Continua |Nivel de educación en cada lugar|
# |Urban|Categórica|Variable valores Sí y No para indicar si la tienda está en una ubicación urbana o rural|
# |US|Categórica |Variable con valores Sí y No para indicar si la tienda está en EE. UU. o no|

# ## 1. Librerias

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, make_scorer


# ## 2. Lectura de datos

# In[3]:


# Descargar el conjunto de datos
get_ipython().system('curl -L -O https://github.com/selva86/datasets/raw/master/Carseats.csv')


# In[4]:


# Leer el dataset
df = pd.read_csv("Carseats.csv")


# In[5]:


# Dimensiones del dataset
df.shape


# In[6]:


# Primeras filas del dataset
df.head()


# ## 3. Preprocesamiento

# In[7]:


# Variables con valores nulos
df.isnull().sum()


# In[8]:


# Generar un target con un valor de corte
df.loc[df["Sales"] >= 8, "High"] = "Yes"
df.loc[df["Sales"] < 8, "High"] = "No"


# ## 4. Analisis Exploratorio de Datos

# In[9]:


# Estadistica descriptiva
df.describe()


# ### 4.1. Analisis Univariado

# In[15]:


# series -> iteritems()
# dataframes -> iterrows()
df.dtypes


# In[11]:


for i in df.dtypes.iteritems():
  print(i)


# In[ ]:


# Ver distribucion de las variables continuas y categoricas
for feature, data_type in df.dtypes.iteritems():
  if data_type != "object":
    fig, ax = plt.subplots(2, 1, figsize=(7, 5), gridspec_kw={"height_ratios": (2, 0.5)})
    fig.suptitle(feature, fontsize=15)
    sns.histplot(data=df, x=feature, kde=True, ax=ax[0])
    ax[0].set(xlabel=None)
    sns.boxplot(data=df, x=feature, ax=ax[1])
  else:
    sns.barplot(x=df[feature].value_counts().index, y=df[feature].value_counts().values)
    plt.suptitle(feature, fontsize=15)
  plt.show()
  print()


# ### 4.2. Analisis Bivariado

# In[ ]:


# Generar matriz de scatter plots
g = sns.pairplot(df)
g.fig.set_size_inches(15, 15)


# ## 5. Modelo

# In[28]:


# Convertir variables categoricas a dummies
df = pd.get_dummies(df, columns=["Urban", "US", "High"], drop_first=True)


# In[29]:


# Conversion de las variable categorica ShelveLoc a numerica
df.loc[df["ShelveLoc"] == "Bad", "ShelveLoc"] = 0
df.loc[df["ShelveLoc"] == "Medium", "ShelveLoc"] = 1
df.loc[df["ShelveLoc"] == "Good", "ShelveLoc"] = 2


# In[30]:


# Revision del tipo de dato de ShelveLoc
df.dtypes


# In[33]:


# Cambiar el tipo de dato
df["ShelveLoc"] = df["ShelveLoc"].astype(int)


# In[34]:


# Se comprueba que se cambio el tipo de dato
df.dtypes


# In[35]:


df.head()


# In[36]:


# Generar matriz de disenho y vector target
X = df.drop(["Sales", "High_Yes"], axis=1)
y = df["High_Yes"]


# In[37]:


# Generar conjunto de datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[39]:


# Generar grilla
tree_grid = {
  "max_depth": np.arange(1, 15, 1)
}


# In[40]:


# Utilizar gridsearch y cross validation para 
clf = GridSearchCV(
  estimator=DecisionTreeClassifier(),
  param_grid=tree_grid,
  cv=5,
  n_jobs=-1)

clf.fit(X_train, y_train)


# In[42]:


# Se muestra la mejor profundidad y el mejor criterio de decision
print("Mejor profundidad:", clf.best_params_["max_depth"])


# In[43]:


# Se entrena el modelo con los mejores parametros identificados
tree = DecisionTreeClassifier(
  max_depth=clf.best_params_["max_depth"],
  random_state=42
)
tree_fit = tree.fit(X_train, y_train)


# In[44]:


# Prediccion
y_pred = tree.predict(X_test)


# # 6. Metricas

# In[46]:


# Se genera la matriz de confusion
cm = confusion_matrix(y_test, y_pred)


# In[47]:


# Se utiliza el metodo de scikit learn para mostrar la matriz de confusion
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tree.classes_)
disp.plot()
plt.show()


# In[ ]:


plt.figure(figsize=(40, 13))
plot_tree(tree_fit, feature_names=X.columns, class_names=["No", "Yes"], filled=True, fontsize=9)
plt.savefig("decision_tree.png", dpi=100)


# In[ ]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
F1 = 2 * ((precision_score(y_test, y_pred) * recall_score(y_test, y_pred)) / ((precision_score(y_test, y_pred) + recall_score(y_test, y_pred))))

comp = pd.DataFrame({"Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [F1]})
comp.index = ["Decision Tree"]


# In[ ]:


comp


# # 6. Fórmulas
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





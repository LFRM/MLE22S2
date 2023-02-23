#!/usr/bin/env python
# coding: utf-8

# # PD10

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import random
import tensorflow as tf

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier


# Se le proporciona la base de datos MNIST que contiene información sobre 70 mil imágenes de dígitos del 0 al 9. Cada imagen está representada por la información de una grilla de 28x28 píxeles con datos entre el 0 y el 255 que representan la claridad del pixel. Además, cuenta con las etiquetas del dígito real que cada imagen representa. Su objetivo es desarrollar un algoritmo que logre identificar acertadamente el dígito correspondiente en base a la información de los píxeles. Para ello se le pide lo siguiente.
# 
# 1) Cargue los datos y realice el preprocesamiento y una visualización de las imágenes\
# 3) Estime un modelo de SVM para clasificar cada imagen según su dígito. Utilice una muestra de 2000 observaciones para train y 2000 para test.\
# 4) Calcule la matriz de confusón del modelo multiclase y las métricas de desempeño (Accuracy,Precision,Recall,F1_Score). Determine cuál es la métrica más indicada para este problema e interprete.\
# 5) Estime un modelo KNN optimizando el hiperparámetro por cross validation y compare las métricas de desempeño entre los modelos.

# ## 1. Preprocesamiento

# In[2]:


(mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = tf.keras.datasets.mnist.load_data()


# In[3]:


print(mnist_x_train.shape)
print(mnist_y_train.shape)
print(mnist_x_test.shape)
print(mnist_y_test.shape)


# In[11]:


pd.set_option("display.max_columns", None)
pd.DataFrame(mnist_x_train[0])


# In[13]:


fig, ax = plt.subplots(2, 3, figsize=(8, 5))
ax[0, 0].imshow(mnist_x_train[0], cmap='gray')
ax[0, 1].imshow(mnist_x_train[1], cmap='gray')
ax[0, 2].imshow(mnist_x_train[2], cmap='gray')
ax[1, 0].imshow(mnist_x_train[3], cmap='gray')
ax[1, 1].imshow(mnist_x_train[4], cmap='gray')
ax[1, 2].imshow(mnist_x_train[5], cmap='gray');
print(mnist_y_train[:6])


# ## 2. Modelo de SVM

# In[19]:


X_train = pd.DataFrame(mnist_x_train.flatten().reshape((60_000, 28 ** 2))).loc[:1999, :]
X_test = pd.DataFrame(mnist_x_test.flatten().reshape((10_000, 28 ** 2))).loc[:1999, :]
y_train = mnist_y_train[:2000]
y_test = mnist_y_test[:2000]


# In[ ]:


svm = SVC(probability=True)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
y_prob = svm.predict_proba(X_test)


# In[ ]:


# Columnas: Prediccion, Filas: Real
pd.DataFrame(confusion_matrix(y_test, y_pred), columns=range(0, 10), index=range(0, 10))


# Como podemos observar, se trata de un problema de clasificación multiclase, distinto a la clasificación binaria a la que estamos acostumbrados. Esto implica que la matriz de confusión debe tomar en cuenta cada una de las categorías distintas y las métricas de desempeño deben calcularse como un promedio a lo largo de todas las clases.

# In[ ]:


print(classification_report(y_test, y_pred, digits=3))


# In[ ]:


print("Accuracy:", np.round(accuracy_score(y_test, y_pred), 3))
print("Average Precision:", np.round(precision_score(y_test, y_pred,average="weighted"), 3))
print("Average Recall:", np.round(recall_score(y_test, y_pred,average="weighted"), 3))
print("Average F1 score:", np.round(f1_score(y_test, y_pred,average="weighted"), 3))
print("Average ROC AUC score:", np.round(roc_auc_score(y_test, y_prob, average="weighted", multi_class="ovr"), 3))

pd.DataFrame([precision_score(y_test,y_pred,average=None),
                recall_score(y_test,y_pred,average=None),
                f1_score(y_test,y_pred,average=None)],
                index=["Precision","Recall","F1 score"]).round(3)


# Podemos constatar que el modelo tiene un algo poder predictivo para todos los dítigos tanto en recall $tp/(tp+fn)$ como en precision $tp/(tp+fp)$. Además, observamos que la categoría en la que se tiene mayor poder predictivo es en el dígito "1" y menor en el dígito "8. Por otro lado, debido a que el dataset está más o menos balanceado a lo largo de las categorías podemos utilizar el accuracy como una medida confiable del desempeño. Por otro lado, como no sabemos si es más importante minimizar los falsos positivos o los falsos negativos, podemos utilizar el f1_score para ponderar ambos criterios.

# ## 3. Modelo KNN

# In[20]:


knn = KNeighborsClassifier()
parameters = {"n_neighbors":range(1,51)}
clf = GridSearchCV(knn,parameters, scoring="f1_weighted", cv=5)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)
clf.best_params_


# In[21]:


pd.DataFrame(confusion_matrix(y_test,y_pred),columns=range(0,10), index=range(0,10))


# In[22]:


print(classification_report(y_test, y_pred, digits=3))


# In[ ]:


print("Accuracy:", np.round(accuracy_score(y_test,y_pred), 3))
print("Average Precision:", np.round(precision_score(y_test,y_pred,average="weighted"), 3))
print("Average Recall:", np.round(recall_score(y_test,y_pred,average="weighted"), 3))
print("Average F1 score:", np.round(f1_score(y_test,y_pred,average="weighted"), 3))
print("Average ROC AUC score:", np.round(roc_auc_score(y_test,y_prob, average="weighted", multi_class="ovr"), 3))

pd.DataFrame([precision_score(y_test, y_pred,average=None),
                recall_score(y_test, y_pred,average=None),
                f1_score(y_test, y_pred,average=None)],
                index=["Precision", "Recall", "F1 score"]).round(3)


# Observando los resultados de la estimación podemos concluir que el modelo SVM tiene un mejor desempeño para predecir la categoría de dígito que el modelo KNN. Podemos comprobar que SVM supera a KNN en todas las métricas.

# In[ ]:





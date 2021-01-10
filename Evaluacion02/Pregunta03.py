import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import sklearn.linear_model as reglin
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# =============================================================================
# Nombre:Guerrero Huayta Renzo Alexis
# 6PDE MACHINE LEARNING CON PYTHON
# EVALUACION DEL MODULO III Y IV 
# =============================================================================

# =============================================================================
# PREGUNTA 3
# =============================================================================

# Cargamos la data
iris = datasets.load_iris()
# Definimos la variable target y las variables predictoras
X4 = iris.data[:, :2]
y4 = iris.target


plt.scatter(X4[:,0], X4[:,1], c=y4, cmap="gist_rainbow")
plt.xlabel("Spea1 Length", fontsize=18)
plt.ylabel("Sepal Width", fontsize=18)

########EJECUTAMOS K MEANS (clusters=3)################

kmeans = KMeans(n_clusters=3).fit(X4)
centroids = kmeans.cluster_centers_
print("Los centroides[x,y] son : \n"+str("1)")+str(centroids[0])+str("\n")+
      str("2)")+str(centroids[1])+str("\n")+
      str("3)")+str(centroids[2]))


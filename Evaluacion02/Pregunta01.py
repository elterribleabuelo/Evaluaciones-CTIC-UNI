import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import sklearn.model_selection as modelSel
import sklearn.linear_model as reglin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# =============================================================================
# Nombre:Guerrero Huayta Renzo Alexis
# 6PDE MACHINE LEARNING CON PYTHON
# EVALUACION DEL MODULO III Y IV 
# =============================================================================

# =============================================================================
# PREGUNTA 1
# =============================================================================

dataframe_adv=pd.read_csv("Advertising.csv",sep=",")
dataframe_adv.columns
dataframe_adv=dataframe_adv.drop(['Unnamed: 0'],axis=1)

#Declarando las variables dependientes e independientes

X=dataframe_adv[:][['TV','radio','newspaper']]
y=dataframe_adv[:][['sales']]

# =============================================================================
# Parte 1
# =============================================================================

X1_train, X1_test, y1_train, y1_test =modelSel.train_test_split(pd.DataFrame(X[:]['TV']),
                                                            y,
                                                            test_size = 0.3,
                                                            random_state = 5)

lm1 = reglin.LinearRegression()
lm1.fit(X1_train, y1_train)
y1_pred = lm1.predict(X1_test)

### GRAFICANDO ###
plt.scatter(y1_test,y1_pred,c="blue")
plt.xlabel("Sales: $y_1$")
plt.ylabel("Prediccion de Sales: $\hat{y}_1$")
plt.title("Sales Vs Sales Predichos: $y_1$ vs $\hat{y}_1$")

# =============================================================================
# Parte 2
# =============================================================================


X2_train, X2_test, y2_train, y2_test =modelSel.train_test_split(X,
                                                            y,
                                                            test_size = 0.3,
                                                            random_state = 5)

lm2 = reglin.LinearRegression()
lm2.fit(X2_train, y2_train)
y2_pred = lm2.predict(X2_test)

### GRAFICANDO ###
plt.scatter(y2_test,y2_pred,c="green")
plt.xlabel("Sales: $y_2$")
plt.ylabel("Prediccion de Sales: $\hat{y}_2$")
plt.title("Sales Vs Sales Predichos: $y_y$ vs $\hat{y}_2$")
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
# PREGUNTA 2
# =============================================================================

dataframe_tit=pd.read_csv("Titanic.csv",sep=",")
dataframe_tit.columns

# =============================================================================
# Parte A
# =============================================================================

print("El dataset Titanic.csv tiene: "+str(dataframe_tit.shape[0])+" filas")

# =============================================================================
# Parte B
# =============================================================================
#Separamos los nombres de las columnas del dtaframe
columns=(np.array([dataframe_tit.columns]))

#Dimensionamos correctamente
columns_dim=((columns[0][:,np.newaxis]).T)[0]

#Encontramos el numero de registros por columna que no esten vacios  
colum_tot=dataframe_tit.count()

#Encontramos la cantidad de elementos null por columna
s=(np.array(891)-np.array([colum_tot]))[0]

resultado=pd.Series(s,index=columns_dim)

print("Las variables con mayor cantidad de elemetos faltantes son: \n",resultado[resultado>0])


# =============================================================================
# Parte C
# =============================================================================
from scipy.stats import norm
dist = norm(loc=np.mean(dataframe_tit.Age), scale=np.std(dataframe_tit.Age))
x = np.linspace(dist.ppf(0.00001),dist.ppf(0.99999),100)


dataframe_tit.Age.hist(bins=14,density=True).set_title("Edad de tripulantes del titanic")
plt.xlabel("Age")
plt.ylabel("Density")
plt.ylim(0, 0.035)
plt.xlim(-10,80)
plt.plot(x,dist.pdf(x),"r-")

# =============================================================================
# Parte D
# =============================================================================

dataframe_tit[:]["SibSp"][1]
dataframe_tit[:]["Parch"]
ViajaSolo=[]
for i in range(len(dataframe_tit)):
    if dataframe_tit[:]["SibSp"][i]+dataframe_tit[:]["Parch"][i]>0:
         ViajaSolo.append(0)
    else:
        ViajaSolo.append(1)
        
ViajaSolo=pd.Series(ViajaSolo,name='ViajaSolo')
# =============================================================================
# PARTE E
# =============================================================================
 
#Agregamos la variable ViajaSolo al dataframe
dataframe_tit=pd.concat([dataframe_tit,ViajaSolo],axis=1)

      
########## Preprocesando los datos ##########

# Creamos un objeto de tipo series almacenando los nombres de las columnas (variables)
tipos = dataframe_tit.columns.to_series()


#Categorizamos las variables 
tipos = tipos.groupby(dataframe_tit.dtypes).groups

# Lista de tipos de datos (variables)
# no numericas
ctext = tipos[np.dtype('object')]
len(ctext)

# Sacando todos los nombres de las columnas 
# definir cnun como una diferencia de conjuntos 
columnas = dataframe_tit.columns
cnum = list(set(columnas)-set(ctext))
len(cnum)

# Completamos los valores vacios : cnum -> media (mean)
for c in cnum:
    mean = dataframe_tit[c].mean()
    dataframe_tit[c] = dataframe_tit[c].fillna(int(mean))
    

# Completamos los valores vacios : ctext -> moda
#.mode()-> primer indice retorna el valor y el segundo la cantidad de veces q aparece
for c in ctext:
    mode = dataframe_tit[c].mode()[0]
    dataframe_tit[c] = dataframe_tit[c].fillna(mode)
    
# Verificamos la existencia de valores na
dataframe_tit.isnull().any().any()

#Generamos las variables dummy de las columnas Embarked y Sex

Embarked_dummy=pd.get_dummies(dataframe_tit["Embarked"],prefix="Embarked")
Sex_dummy=pd.get_dummies(dataframe_tit["Sex"],prefix="Sex")

#Unimos las variables tipo dummy al dataset original
dataframe_tit=pd.concat([dataframe_tit,Embarked_dummy],axis=1)
dataframe_tit=pd.concat([dataframe_tit,Sex_dummy],axis=1)


#Definiendo las variables dependientes e independientes
Xlog=dataframe_tit[:][['Age','ViajaSolo','Pclass','Embarked_C','Embarked_Q','Embarked_S',
                       'Sex_female','Sex_male']]
ylog=pd.DataFrame(dataframe_tit[:][['Survived']])

#Estandarizamos los datos
sc=StandardScaler()
Xlog=pd.DataFrame(sc.fit_transform(Xlog))

X3_train, X3_test, y3_train, y3_test =modelSel.train_test_split(Xlog,
                                                            ylog,
                                                            test_size = 0.3,
                                                            random_state = 5)

modelLog = reglin.LogisticRegression()
modelLog.fit(X3_train,y3_train)
y3_pred = modelLog.predict(X3_test)
precision=modelLog.score(Xlog,ylog)


print("Datos de entrenamiento:")
print(np.array(y3_test[:]["Survived"]))

print("Datos obtenidos en la prediccion:")
print(y3_pred)

print("La precision media del modelo logístico es:", precision)


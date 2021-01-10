import sympy as sp
import numpy as np
import scipy.optimize as spot
import matplotlib.pyplot as plt
import pandas as pd
import random

### Alumno: Renzo Alexis Guerrero Huayta ###
### Evaluacion del Modulo II --6PDE Machine Learning con Python-- #


####################  PROBLEMA 1  #########################

R=float(input("Ingrese el rendimiento medio de la cartera:"))
Rf=float(input("Ingrese la tasa media libre de riesgo:"))
desv_est=float(input("Ingrese el riesgo de la cartera:"))


def ratio(a,b,c):
    sharpe=(a-b)/c
    return sharpe
print("El ratio de sharpe es:",ratio(R,Rf,desv_est))





####################  PROBLEMA 2  #########################

##### A) RAICES CON SIMPY #####



x = sp.symbols('x ')
f = x**4-22*(x**2)+x+114
solucion=sp.solve(f,x)
type(solucion)

for i in range(len(solucion)):
    print(f"Raiz {i+1}:",solucion[i])
    
    
    
##### B) RAICES CON SCIPY ######
    
def f(x):
    y=x**4-22*(x**2)+x+114
    return y
raices=spot.fsolve(f,[2.0,4.0,-2.0,-4.0])

for i in range(len(raices)):
    print(f"Raiz {i+1}:",raices[i])
    
##### C) MINIMOS CON SCIPY #####

min1 = spot.minimize(f,2.0)
min2 = spot.minimize(f,-2.0)
print("Los minimos del polinomio son:"+str(min1["fun"])+"  y  "+str(min2["fun"]))




####################  PROBLEMA 3  #########################


def gen_rnd(n):
    List1 = []
    yi = int(np.random.randint(1,124578,1))
    for i in range(n):
        yi = (124578*yi+1)%(125)
        List1.append(yi)
    return List1
 
List1 = gen_rnd(500)



###### A) ELEMENTOS UNICOS #####

def unic(List):
    List2=[]
    elementos_unicos=set(List)
    List2=list(elementos_unicos)
    return List2

List2=unic(List1)

print("La lista que contiene los elementos unico es:",List2)




##### B) ELEMENTOS MULTIPLOS DE 7 O 13 #####

def mult(Lista):
    List3=[]
    for i in range(len(Lista)):
        if Lista[i]%7==0 or Lista[i]%13==0 :
            List3.append(Lista[i])
    return List3

List3=mult(unic(List1))

print("La lista con elementos multiplos de 7 o 13 son:",List3)



##### C) PRIMO O NO PRIMO #####

def primo(List2):
    a=[]
    b=[]
    dic_primos=dict()
    for i in range(len(List2)):
        if List2[i]<2:
            b.append(List2[i])
            a.append('False')
        else :
            valor=range(2,List2[i])
            contador=0
            for n in valor:
                if List2[i]%n==0:
                    contador=contador+1
            
            if contador>0:
                b.append(List2[i])
                a.append('False')
            else :
                 b.append(List2[i])
                 a.append('True')
    dic_primos=dict(zip(b,a))
    return dic_primos
        
            
dic_primos=primo(List2)
print("El diccionario es el siguiente:",dic_primos)
        
    

            
####################  PROBLEMA 4  #########################

Caracteres = [chr(i) for i in range(128)][33:-1]

Numeros=Caracteres[15:25]
Mayusculas=Caracteres[32:58]
Minusculas=Caracteres[64:90]
Simbol1=Caracteres[0:15]
S1=set(Simbol1)
Simbol2=Caracteres[25:32]
S2=set(Simbol2)
Simbol3=Caracteres[58:64]
S3=set(Simbol3)
Simbol4=Caracteres[90:94]
S4=set(Simbol4)

S5=S1|S2|S3|S4
Simbol5=list(S5)

def psw(n):
    nu=random.sample(Numeros,2)
    Num=set(nu)
    ma=random.sample(Mayusculas,1)
    May=set(ma)
    mi=random.sample(Minusculas,n-6)
    Min=set(mi)
    si=random.sample(Simbol5,3)
    Sim=set(si)
    contra=Num|May|Min|Sim
    Password=list(contra)
    return Password
    
rpta=1
while rpta==1:
    n=int(input("Ingrese la longitud de la contraseña:"))
    if n>=9:
        print ("La contraseña generada es: "+''.join(psw(n)))
        rpta=int(input("""Para obtener una nueva contraseña presione 1 , para salir presione 2 :"""))
    else :
        print("La longitud de la contraseña debe ser mayor a 9")


        
        
####################  PROBLEMA 5  #########################
        
data_TX = pd.read_csv("nombres_TX.TXT",sep=",")
data_CA = pd.read_csv("nombres_CA.txt",sep=",")



##### A) BEBES NACIDOS POR AÑO EN CADA CIUDAD #####

def num_bebes(n):
    Sum_CA=0
    Sum_TX=0
    List_num_bebes=[]
    for i in range(len(data_CA.index)):
        if int(data_CA['year'][i])==n:
            Sum_CA=Sum_CA+int(data_CA['Num'][i])
    List_num_bebes.append(Sum_CA)
            
    for k in range(len(data_TX.index)):
        if int(data_TX['year'][k])==n:
            Sum_TX=Sum_TX+int(data_TX['Num'][k])
    List_num_bebes.append(Sum_TX)       
    
    return List_num_bebes


        


print("---Calcular el numero de bebes nacidos por año en cada ciudad---")
n=int(input("Ingrese el año:"))
if n in range(int(data_CA['year'].describe()['min']),int(data_CA['year'].describe()['max']+1)):
    List_num_bebes=num_bebes(n)
    print(f"""El numero de bebes nacidos en el año {n} en la ciudad CA  es:""",List_num_bebes[0] )
    print(f"""El numero de bebes nacidos en el año {n} en la ciudad TX  es:""",List_num_bebes[1] )
else:
    print("El año que ingreso no se encuentra registrado.")
    

    
#### B)CUANTOS BEBES (SEX=F) NACIERON ENTRE UN RANGO DE AÑOS ####
    
data_femenino_CA=data_CA.loc[data_CA['Sex'] == "F"]
data_femenino_TX=data_TX.loc[data_TX['Sex'] == "F"]


def f1(year_min,year_max):
    List_fem=[]
    j=0
    for i in range(len(data_femenino_CA.index)):
        if data_femenino_CA['year'][i] in range(year_min,year_max+1):
            j=j+1
    List_fem.append(j)
    k=0
    for i in range(len(data_femenino_TX.index)):
        if data_femenino_TX['year'][i] in range(year_min,year_max+1):
            k=k+1
    List_fem.append(k)
    
    return List_fem

print("-----Cuantos bebes hay por rango de años----")
year_min=int(input("Desde el año:"))
if year_min in range(int(data_CA['year'].describe()['min']),int(data_CA['year'].describe()['max']+1)):
    year_max=int(input("Hasta el año:"))
    if year_max in range(int(data_CA['year'].describe()['min']+1),int(data_CA['year'].describe()['max']+1)):
        List_fem=f1(year_min,year_max)    
        print("El numero de bebes con sexo femenino nacidos en CA son:",List_fem[0])
        print("El numero de bebes con sexo femenino nacidos en TX son:",List_fem[1])
        print("Total:",List_fem[0]+List_fem[1])
    else:
        print("""El año que ingreso no se encuentra registrado,
              ademas este debe ser mayor al año de inicio""")

else:
    print("El año que ingreso no se encuentra registrado")



##### C)3 NOMBRES MAS COMUNES EN CADA CIUDAD ######

Name_CA=data_CA["Name"].value_counts()
Name_TX=data_TX["Name"].value_counts()

First3_CA=Name_CA.index[0:3]
First3_TX=Name_TX.index[0:3]


def tres_primeros(ciudad):
    if ciudad=='CA':
        return print("Los 3 nombres mas comunes en la ciudad CA son:",First3_CA[0],First3_CA[1],First3_CA[2])
    elif ciudad=='TX':
        return print("Los 3 nombres mas comunes en la ciudad TX son:",First3_TX[0],First3_TX[1],First3_TX[2])
    else :
        return print("Esta ciudad no existe...")

        
### TRES PRIMEROS EN LA CIUDAD CA ###
tres_primeros(data_CA['State'][0])


### TRES PRIMEROS EN LA CIUDAD TX ###
tres_primeros(data_TX['State'][0])




##### D) SEGUNDO NOMBRE MAS POPULAR POR DECADA ######


def f4(n):
    List_final=[]
    
    
    data_TX_dec1=data_TX[(data_TX.year>=n) & (data_TX.year<=n+10) ]
    data_tx_dec1_Seg=data_TX_dec1["Name"].value_counts()
    d=data_tx_dec1_Seg.tolist()
    ##Utilizamos la funcion unic() definida en la parte a del problema 3
    ListU=unic(d)
    w=max(ListU)
    ListU.remove(w) 
    cont=0
    
    for i in range(0,len(data_tx_dec1_Seg.values)):
        if data_tx_dec1_Seg.values[i]>=data_tx_dec1_Seg.values[0]:
            cont=cont+1
    
    
    List_indices_TX=[]
    for j in range(0,100):
        if data_tx_dec1_Seg.values[cont+j-1]==max(ListU):
            List_indices_TX.append(cont+j-1)

    List_nombres_TX=[]
    for k in range(min(List_indices_TX),max(List_indices_TX)+1):
        List_nombres_TX.append(data_tx_dec1_Seg.index[k])
        
    List_final.append(List_nombres_TX)
    
    
    data_CA_dec1=data_CA[(data_CA.year>=n) & (data_CA.year<=n+10) ]
    data_ca_dec1_Seg=data_CA_dec1["Name"].value_counts()
    e=data_ca_dec1_Seg.tolist()
    ##Utilizamos la funcion unic() definida en la parte a del problema 3
    List_CA=unic(e)
    y=max(List_CA)
    List_CA.remove(y) 
    cont3=0
    
    for l in range(0,len(data_ca_dec1_Seg.values)):
        if data_ca_dec1_Seg.values[l]>=data_ca_dec1_Seg.values[0]:
            cont3=cont3+1
    
    
    List_indices_CA=[]
    for m in range(0,100):
        if data_ca_dec1_Seg.values[cont3+m-1]==max(List_CA):
            List_indices_CA.append(cont3+m-1)

    List_nombres_CA=[]
    for n in range(min(List_indices_CA),max(List_indices_CA)+1):
        List_nombres_CA.append(data_ca_dec1_Seg.index[n])
        
    List_final.append(List_nombres_CA)
    
        
    return (List_final)



año=int(input("Ingrese el año:"))  
if año in range(1910,2019,11):      
    print(f"1)El segundo nombre/s mas populares en la descada de {año}-{año+10} en la ciudad {data_TX.State[0]} son :" +','.join(f4(año)[0]))
    
    print(f"2)El segundo nombre/s mas populares en la descada de {año}-{año+10} en la ciudad {data_CA.State[0]} son :" +','.join(f4(año)[1]))

else:
    print("""Debe ingresar un año en el que comienza una decada
          (1910,1921,1932,1943,1954,1965,1976,1987,1998,2009)""")

    
        
    
    
####################  PROBLEMA 6  #########################
    
cherry = pd.read_csv("cherry.csv", sep=";")


##### A) CANTIDAD DE FILAS DEL DATASET #####

num_filas=len(cherry.index)
print(f"El dataset tiene {num_filas} filas")



##### B) DIAGRAMA DE DISPERSION DE CADA VARIABLE #####


#Grafico 1 Girth vs Volumen#

fig,ax=plt.subplots()
ax.scatter(cherry.Girth,cherry.Volume,color='r')
plt.ylabel('Volume')
plt.xlabel('Girth')
plt.title('Grafico de dispersión')
plt.show()



#Grafico 2 Height vs Volume#

fig,ax=plt.subplots()
ax.scatter(cherry.Height,cherry.Volume,color='b')
plt.ylabel('Volume')
plt.xlabel('Height')
plt.title('Grafico de dispersión')
plt.show()



#####  C) BOXPLOT PARA CADA VARIABLE  #####


## 1)Boxplot para Girth ##

plt.boxplot(cherry.Girth)
plt.ylabel('Girth')
plt.title('BOXPLOT PARA GIRTH')
plt.show()


## 2)Boxplot para Height ##

plt.boxplot(cherry.Height)
plt.ylabel('Height')
plt.title('BOXPLOT PARA HEIGHT')
plt.show()


## 3)Boxplot para Volume ##

plt.boxplot(cherry.Volume)
plt.ylabel('Volume')
plt.title('BOXPLOT PARA VOLUME')
plt.show()

# NOTA:Como se ve en el Boxplot de Volumen el valor atipico es 77#

for i in range(0,len(cherry)):
    if cherry.Volume[i]==77:
        g=cherry.Girth[i]
        h=cherry.Height[i]

print(f"""Los valores de Girth y Height para el outlier de Volume son: 
          {g} y {h} respectivamente""")
        





            
            
        


        
        

    


    
    


            
            
    
    

             
     
    



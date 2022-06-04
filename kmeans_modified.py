import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import math
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import combinations
start_time1 = datetime.now()
df1=pd.read_excel("Iris.xls")
klist=[]
distlist=[]
XB=[]
def iterd(a):
    a=round(math.sqrt(a))
    if (a%2)==0:
        a=a+1
        return(a)
    else:
        return(a)
for k in range(2,iterd(len(df))):
    df=shuffle(df1)
    df=df.drop("iris",axis=1)
    df["grp"]=0
    grp=df.sample(n=k,random_state=68)
    grp["value"]=0
    i=0
    for index,row in grp.iterrows():
        row["grp"]=i
        grp.loc[index]=row
        i=i+1
    
    for index1 in df.index:
        for index in grp.index:
            grp.at[index,"value"]=math.sqrt(((df.at[index1,"sepal length"]-grp.at[index,"sepal length"])**2)+((df.at[index1,"sepal width"]-grp.at[index,"sepal width"])**2)+((df.at[index1,"petal length"]-grp.at[index,"petal length"])**2)+((df.at[index1,"petal width"]-grp.at[index,"petal width"])**2))
        grp=grp.sort_values(by="value")
        df.at[index1,"grp"]=grp.head(1)["grp"]
    
    for jk in range(20):
        
        g=df.groupby("grp")
        for s,r in g:
            grp["sepal length"][grp["grp"]==s]=df["sepal length"][df["grp"]==s].mean()
            grp["sepal width"][grp["grp"]==s]=df["sepal width"][df["grp"]==s].mean()
            grp["petal length"][grp["grp"]==s]=df["petal length"][df["grp"]==s].mean()
            grp["petal width"][grp["grp"]==s]=df["petal width"][df["grp"]==s].mean()
        value=0
        for index1 in df.index:
            for index in grp.index:
                grp.at[index,"value"]=math.sqrt(((df.at[index1,"sepal length"]-grp.at[index,"sepal length"])**2)+((df.at[index1,"sepal width"]-grp.at[index,"sepal width"])**2)+((df.at[index1,"petal length"]-grp.at[index,"petal length"])**2)+((df.at[index1,"petal width"]-grp.at[index,"petal width"])**2))
            grp=grp.sort_values(by="value")
            value=value+float(grp.head(1)["value"])**2          
            df.at[index1,"grp"]=grp.head(1)["grp"]
    klist.append(k)
    distlist.append(value)
    print(df["grp"].value_counts())
    end_time1 = datetime.now()
    print('Duration: {}'.format(end_time1 - start_time1))
    perm=list(combinations(grp.index,2))
    small=999999999
    for ki in perm:
        temp=((df.at[ki[0],"sepal length"]-grp.at[ki[1],"sepal length"])**2)+((df.at[ki[0],"sepal width"]-grp.at[ki[1],"sepal width"])**2)+((df.at[ki[0],"petal length"]-grp.at[ki[1],"petal length"])**2)+((df.at[ki[0],"petal width"]-grp.at[ki[1],"petal width"])**2)
        if(small>temp):
            small=temp
    XB.append(value/(small*len(df))) 
sns.lineplot(klist,XB)
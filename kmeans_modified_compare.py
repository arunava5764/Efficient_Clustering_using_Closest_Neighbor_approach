import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import math
import seaborn as sns
from scipy.spatial import distance
import matplotlib.pyplot as plt
def iterd(a):
    a=round(math.sqrt(a))
    if (a%2)==0:
        a=a+1
        return(a)
    else:
        return(a)
df=pd.read_excel("Iris.xls")
df=shuffle(df)
klist=[]
distlist=[]
# =============================================================================
# G1=pd.DataFrame()
# G2=pd.DataFrame()
# G3=pd.DataFrame()
# =============================================================================
df=df.drop("iris",axis=1)
for k in range(2,iterd(len(df))):
    df["grp"]=0
    grp=df.sample(n=k,random_state=56)
    grp["value"]=0
    i=0
    for index,row in grp.iterrows():
        row["grp"]=i
        grp.loc[index]=row
        i=i+1
    
    for index1,row1 in df.iterrows():
        for index,row in grp.iterrows():
            row["value"]=distance.euclidean(list(row.drop(["grp","value"])),list(row1.drop(["grp"])))
            grp.loc[index]=row
        grp=grp.sort_values(by="value")
        row1["grp"]=grp.head(1)["grp"]
        df.loc[index1]=row1
    for jk in range(20):
    
        g=df.groupby("grp")
        for s,r in g:
            grp["sepal length"][grp["grp"]==s]=df["sepal length"][df["grp"]==s].mean()
            grp["sepal width"][grp["grp"]==s]=df["sepal width"][df["grp"]==s].mean()
            grp["petal length"][grp["grp"]==s]=df["petal length"][df["grp"]==s].mean()
            grp["petal width"][grp["grp"]==s]=df["petal width"][df["grp"]==s].mean()
        dist=0
        for index1,row1 in df.iterrows():
            for index,row in grp.iterrows():
                row["value"]=distance.euclidean(list(row.drop(["grp","value"])),list(row1.drop(["grp"])))
                grp.loc[index]=row
            grp=grp.sort_values(by="value")
            row1["grp"]=grp.head(1)["grp"]
            dist=dist+float(grp.head(1)["value"]**2)
            df.loc[index1]=row1
    klist.append(k)
    distlist.append(dist)
print(klist)
print(distlist)
plt.plot(klist,distlist)
plt.show()

        
        
 
  



    
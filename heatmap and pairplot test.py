import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



df = pd.read_csv("train.csv")
columnsofinterest = ["SalePrice"]
datacolumns = df[columnsofinterest]
#print(datacolumns.describe())
corr=df.corr()["SalePrice"]
#corrilation
print(corr[np.argsort(corr, axis=0)[::-1]])


#heatmap with corrolation
plt.figure(figsize=(7,4)) 
columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']
sns.heatmap(df[columns].corr(),annot=True,cmap='cubehelix_r') #draws  heatmap with input as the correlation matrix calculted by(iris.corr())
plt.show()

#pairplot
sns.set()
columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']
sns.pairplot(df[columns],size = 2 ,kind ='scatter')
plt.show()
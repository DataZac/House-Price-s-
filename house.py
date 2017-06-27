import math
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
from sklearn.linear_model import LinearRegression

#(T) .shape, #rehape
#__________________Settings___________________
pd.set_option('display.max_columns', 500)
plt.style.use('ggplot')
#__________________Functions__________________
def calcrmse(sumofresids, n):#Equal name: rms-Deviation. RMSE=Variance(of y-hat) + Bias (of y-hat and y)
    return [math.sqrt(sumofresids/n),]
def makeOutputFile:

#__________________Main
df_train = pd.read_csv('train.csv', sep=',', index_col=0)

df_test =  pd.read_csv('test.csv', sep=',', index_col=0)


##################
X=df_train['LotArea'].apply(lambda x: x*0.09290304).reshape(len(df_train),1) #Regr method needs (rows, column(s))-array shape. Scale to square meters

y=df_train['SalePrice'].reshape(len(df_train),1)

Xtest=df_test['LotArea'].reshape(len(df_test),1)



linreg=LinearRegression()

linreg.fit(X,y)

print("Intercept:",linreg.intercept_)

print("Coefficients:", linreg.coef_)

print("Residues:", linreg.residues_)#Sum of residuals. Squared Euclidean 2-norm for each target passed during the fit

print("RMSE:", calcrmse(linreg.residues_, len(df_train)))


c=linreg.predict(X)
plt.figure()
plt.scatter(X,y, color='black')
plt.plot(X, c, color='red')
plt.show()

dferror=pd.DataFrame(data=c, columns=['Predicted'])
dferror['SalePrice']=y
dferror['Residues']=dferror.Predicted-dferror.SalePrice
dferror['LogResidues']=(dferror.Predicted.apply(lambda x: np.log(x)))-(dferror.SalePrice.apply(lambda x: np.log(x)))
squarederrorsum=dferror.Residues.apply(lambda x: x*x).sum()
squarederrorsumofresidues=dferror.LogResidues.apply(lambda x: x*x).sum()
print ("RMSE by Hand:", calcrmse(squarederrorsum, len(dferror)))
print ("LOGRMSE by Hand:", calcrmse(squarederrorsumofresidues, len(dferror)))
fig=plt.figure()
fig.add_subplot(121).scatter(X,dferror.Residues, color='red')
fig.add_subplot(122).scatter(X,dferror.LogResidues, color='blue')
plt.show()
"""
ctest=linreg.predict(Xtest)
plt.figure()
plt.scatter(Xtest, ctest, color='red')
plt.show()



###############


ssd0=df_train.corr(method='pearson').SalePrice.sort_values(ascending=False)

print(ssd0)

ssd0=df_train.corr(method='spearman').SalePrice.sort_values(ascending=False)

print(ssd0)



df_train.plot.scatter('EnclosedPorch','SalePrice')
df_train.loc[df_train.EnclosedPorch> 0,:].corr().SalePrice.sort_values(ascending=False)

#feature_list=list(df_train)[:3]#drop SalePrice
feature_list=['GrLivArea','LotArea','EnclosedPorch']
pp = sns.pairplot(data=df_train,
                  y_vars=['SalePrice'],
                  x_vars=feature_list)

#__________________________

X=df_train['GrLivArea'].reshape(len(df_train),1) #Regr method needs (rows, column(s))-array shape. Scale to square meters

y=df_train['SalePrice'].reshape(len(df_train),1)

linreg=LinearRegression()

linreg.fit(X,y)

print("Intercept:",linreg.intercept_)

print("Coefficients:", linreg.coef_)

print("Residues:", linreg.residues_)

print("RMSE:", calcrmse(linreg.residues_, len(df_train)))

c=linreg.predict(X) # Predicting the test data gives the empirical fit
plt.figure()
plt.scatter(X,y, color='black')
plt.plot(X, c, color='red')
plt.show()

c=linreg.predict(Xtest)

plt.figure()
plt.scatter(Xtest, ctest, color='red')
plt.show()


"""

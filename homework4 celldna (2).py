
# coding: utf-8

# In[762]:


import matplotlib as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
import pandas as pd
from pylab import rcParams
import seaborn as sb
from collections import Counter
import statsmodels.formula.api as smapi
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn import preprocessing
import matplotlib as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale
import pandas as pd
from pylab import rcParams
import seaborn as sb
from collections import Counter
import statsmodels.formula.api as smapi
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn import preprocessing
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import lasso_path, lars_path, Lasso, enet_path


# In[763]:



dataset = "Documents\\CellDNA1.csv"
DNA = pd.read_csv(dataset)
dataset = np.genfromtxt("Documents\\CellDNA1.csv", delimiter = ',')


# In[764]:


DNA.head()


# In[765]:


DNA.shape


# In[766]:


DNA.loc[:, ["222","31.18918919", "40.34234234", "35.57908668","8.883916969","0.968324558", "-80.11367302", "222.1","1","16.81247093","0.816176471", "0.578125", "78.591", "0"]]


# In[767]:


sb.pairplot(DNA)


# In[768]:


df = pd.read_csv("Documents\\CellDNA1.csv",sep=",")


# In[769]:


X_scaled = preprocessing.scale(DNA)


# In[770]:


X_scaled


# In[771]:


from scipy.stats import zscore
nz = DNA.apply(zscore)


# In[772]:


nz


# In[773]:


X1= X_train = DNA.loc[:, ["222","31.18918919", "40.34234234", "35.57908668","8.883916969","0.968324558", "-80.11367302", "222.1","1","16.81247093","0.816176471", "0.578125", "78.591"]]


# In[774]:


X1


# In[775]:


from scipy.stats import zscore
XT = X1.apply(zscore)


# In[776]:


XT


# In[777]:


X_scaled = preprocessing.scale(X_train)


# In[778]:


print(X_scaled, '\n')
print(X_scaled.mean(axis=0), X_scaled.std(axis=0))


# In[779]:


from scipy.stats import zscore
Y1 = X_train.apply(zscore)


# In[780]:


Y1


# In[781]:


Y = DNA.loc[:, ['0']]


# In[782]:


Y


# In[783]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import lasso_path, lars_path, Lasso, enet_path


# In[784]:


X = X1 
Y = Y



# In[785]:


Y = dataset[:,2];
X = dataset[:,:13];  
X[np.isnan(X)] = 0 


# In[786]:


print("Regularization path using lars_path")    


# In[787]:


alphas1, active1, coefs1= lars_path(X, Y, method='lasso', verbose=True)




# In[788]:


print("Regularization path using lars_path")  


# In[789]:


eps= 5e-6


# In[790]:


alphas2, coefs2, _= lasso_path(X, Y, eps)      


# In[791]:


print("ONE regularization using Lasso")


# In[792]:


clf = Lasso(fit_intercept=False, alpha=1.3128)


# In[793]:


clf.fit(X, Y)


# In[794]:


print(clf.intercept_, clf.coef_)


# In[795]:


clf.fit(X, Y)
print(clf.intercept_, clf.coef_)


# In[796]:


#log_alphas= -np.log10(model.alphas_)
#ax = plt.gca()
plt.plot(model.alphas_, model.coef_path_.T)
#plt.axvline(log_alphas, linestyle='--', color='k', label='alpha CV')


plt.xlabel('alpha')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.show()

fig, ax = plt.subplots(figsize=(30,25))
xx = np.sum(np.abs(coefs1.T), axis=1)
plt.plot(xx, coefs1.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed',  label='alpha CV')

plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LARS Path')

plt.legend()


# In[797]:


#log_alphas= -np.log10(model.alphas_)
#ax = plt.gca()
plt.plot(model.alphas_, model.coef_path_.T)
#plt.axvline(log_alphas, linestyle='--', color='k', label='alpha CV')
xx = np.sum(np.abs(coefs1.T), axis=1)

plt.plot(xx, coefs1.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed',  label='alpha CV')

plt.xlabel('alpha')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.show()



# In[798]:


xx


# In[799]:


#alphas,coefs, dual_gaps=model.path(X_train, y)
#coefs=coefs.reshape(11,100)
#result=pd.DataFrame(coefs.T,index=alphas,columns=X_train.columns.values.tolist())
#result.head()
xx = np.sum(np.abs( model.coef_path_.T ), axis=1)
plt.plot(xx, model.coef_path_.T)
plt.plot(xx, model.coef_path_.T)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
ymin, ymax = plt.ylim()
#plt.vlines(alphas, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.show()


# In[800]:


from sklearn.linear_model import Lasso
for aa in np.arange(0,1.1, 0.01):
    clf = Lasso(alpha = aa)
    clf.fit(X1, Yz)
    print(aa, clf.coef_, clf.intercept_)


# In[801]:


from sklearn.model_selection import cross_val_score


from sklearn.cross_validation import cross_val_score



clf = linear_model.Lasso()
scores = cross_val_score(clf, X, Y, cv=10)


# In[802]:


scores


# In[803]:


scores.mean()


# In[804]:


scores.std()


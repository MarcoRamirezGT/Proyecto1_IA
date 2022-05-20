from sklearn.cluster import KMeans

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib.patches import Ellipse

from sklearn.mixture import GaussianMixture





dataset = pd.read_csv('Fraud.csv')
df2 = pd.DataFrame(dataset)


print(df2)

X = df[['amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest']]
kmeans= KMeans(4, random_state=0)
print (X)
Y = df2[['isFraud']]
print(Y)





# ## metodo para determinar el numero de clusters
# N = np.arange(1,10)
# state= 0
# gmm = [ GaussianMixture( n, covariance_type ="full", random_state=state).fit(X)  for n in N]
# for model in gmm :
# 	print(model.bic(X))

# plt.plot(N, [model.bic(X) for model in gmm] , "o-", label ="BIC")
# plt.xlabel("# de clusters")
# plt.ylabel("BIC score")
# plt.title("Gráfico de Codo Gaussian Mixture models ")
# plt.savefig('codo Gaussian mixture models.png')
# plt.show()


from sklearn.cluster import KMeans

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from matplotlib.patches import Ellipse

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import time

# Para medir el tiempo
start_time = time.time()



dataset = pd.read_csv('Fraud.csv')
df2 = pd.DataFrame(dataset)

## se necesitan tener cantidades similares de fraudulentos y no fraudulentos
## se tienen 8213 registros fraudulentos
## el 70 prociento de los registros fraudulentos seria 5750



df_notFraud = df2[df2['isFraud'] == 0]
df_notFraudTrainig = df2.head(5750)


df_fraud = df2[df2['isFraud'] == 1]
df_FraudTraining= df_fraud.head(5750)

df_training = pd.concat([df_FraudTraining, df_notFraudTrainig])


X = df_training[['amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest']]
print (X)
Y = df_training['isFraud']
print(Y)

print(df_training)


scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)


kmeans= KMeans(4, random_state=0)
kmeans.fit(X)

predicted_y = kmeans.fit_predict(X)

print("--- %s seconds ---" % (time.time() - start_time))



# kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 42,}
# # A list holds the SSE values for each k
# sse = []
# for k in range(1, 11):
#         kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
#         kmeans.fit(scaled_features)
#         sse.append(kmeans.inertia_)



# plt.style.use("fivethirtyeight")
# plt.plot(range(1, 11), sse)
# plt.xticks(range(1, 11))
# plt.xlabel("Number of Clusters")
# plt.ylabel("SSE")
# plt.show()



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

print("--- %s seconds ---" % (time.time() - start_time))
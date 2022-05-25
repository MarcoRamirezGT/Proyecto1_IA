
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time


# Para medir el tiempo
start_time = time.time()


#ESTADISTICO DE HOPKINS
# X_scale=sklearn.preprocessing.scale(X)
# print(pyclustertend.hopkins(X,len(X)))

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
#print (X)
Y = df_training['isFraud']
#print(Y)

#print(df_training)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3)


gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
failure= ((y_test != y_pred).sum()) / (X_test.shape[0])
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
print( " Accuracy" , 1-failure)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(X)
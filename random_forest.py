from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
# Import scikit-learn dataset library
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from IPython.display import Image
from six import StringIO
import pydotplus
import os
import time
start_time = time.time()
df = pd.read_csv('Fraud.csv')
df = pd.read_csv('Fraud.csv')

df_2 = df[df['isFraud'] == 0]
df_3 = df_2.head(8213)

df_1 = df[df['isFraud'] == 1]

df = pd.concat([df_1, df_3])

X = df[['amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest']]
y = df['isFraud']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)  # 70% training and 30% test

# Import Random Forest Model

# Create a Gaussian Classifier
clf = RandomForestClassifier(
    n_estimators=100, random_state=0, criterion="entropy", max_depth=5)

# Train the model using the training sets y_pred=clf.predict(X_test)
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Import scikit-learn metrics module for accuracy calculation
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf.estimators_[0], out_file=dot_data,
                filled=True, rounded=True,
                special_characters=True, feature_names=['amount', 'oldbalanceOrg', 'newbalanceOrig',
                                                        'oldbalanceDest', 'newbalanceDest'], class_names=['No fraude', 'Fraude'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('RandomForest.png')
Image(graph.create_png())
print("--- %s seconds ---" % (time.time() - start_time))

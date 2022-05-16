from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
# Import scikit-learn dataset library
from sklearn import datasets
from sklearn.model_selection import train_test_split

df = pd.read_csv('Fraud.csv')

print(df)
#print(df[['amount', 'oldbalanceOrg']])
X = df[['amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest']]
y = df['isFraud']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3)  # 70% training and 30% test

# Import Random Forest Model

# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)

# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Import scikit-learn metrics module for accuracy calculation
# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

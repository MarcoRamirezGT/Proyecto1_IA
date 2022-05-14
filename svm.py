from re import S
import warnings
import pandas as pd
import numpy as np
# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from mlxtend.plotting import plot_decision_regions

# Preprocesado y modelado
# ==============================================================================
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Configuración matplotlib
# ==============================================================================
plt.rcParams['image.cmap'] = "bwr"
#plt.rcParams['figure.dpi'] = "100"
plt.rcParams['savefig.bbox'] = "tight"
style.use('ggplot') or plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
warnings.filterwarnings('ignore')

df = pd.read_csv('Fraud.csv')
print(df.head(3))

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df.oldbalanceOrg, df.newbalanceOrig, c=df.isFraud)
ax.set_title("Datos ESL.mixture")
plt.show()
plt.savefig('Datos ESL')

# División de los datos en train y test
# ==============================================================================
s = 'isFraud', 'step', 'type', 'nameOrig', 'nameDest', 'isFlaggedFraud'
X = df.drop(['isFraud', 'step', 'type', 'nameOrig',
            'nameDest'], axis=1)
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y.values.reshape(-1, 1),
    train_size=0.8,
    random_state=1234,
    shuffle=True
)
# Creación del modelo SVM lineal
# ==============================================================================
modelo = SVC(C=100, kernel='linear', random_state=123)
modelo.fit(X_train, y_train)

# Representación gráfica de los límites de clasificación
# ==============================================================================
# Grid de valores
x = np.linspace(np.min(X_train.X1), np.max(X_train.X1), 50)
y = np.linspace(np.min(X_train.X2), np.max(X_train.X2), 50)
Y, X = np.meshgrid(y, x)
grid = np.vstack([X.ravel(), Y.ravel()]).T

# Predicción valores grid
pred_grid = modelo.predict(grid)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(grid[:, 0], grid[:, 1], c=pred_grid, alpha=0.2)
ax.scatter(X_train.X1, X_train.X2, c=y_train, alpha=1)

# Vectores soporte
ax.scatter(
    modelo.support_vectors_[:, 0],
    modelo.support_vectors_[:, 1],
    s=200, linewidth=1,
    facecolors='none', edgecolors='black'
)

# Hiperplano de separación
ax.contour(
    X,
    Y,
    modelo.decision_function(grid).reshape(X.shape),
    colors='k',
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=['--', '-', '--']
)

ax.set_title("Resultados clasificación SVM lineal")

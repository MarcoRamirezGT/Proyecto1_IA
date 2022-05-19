import pandas as pd

df = pd.read_csv('Fraud.csv')

df_2 = df[df['isFraud'] == 0]
df_3 = df_2.head(8213)

df_1 = df[df['isFraud'] == 1]

df = pd.concat([df_1, df_3])

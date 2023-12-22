import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.cluster import KMeans

df = pd.read_csv('C:\\Users\\deela\\Downloads\\Mall_Customers.csv')
print(df)

print(df.describe())

# Machine Learning


# Label Encoding
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
df.Genre = enc.fit_transform(df.Genre)

df.drop('CustomerID', axis=1, inplace=True)

'''ssd = []
for i in range(1,11):
  Kmodel = KMeans(n_clusters=i, n_init=15,max_iter=500)
  Kmodel.fit(df)
  ssd.append(Kmodel.inertia_)

print(ssd)'''

Kmodel = KMeans(n_clusters=6)
Kmodel.fit(df)

prediction = Kmodel.predict(df)
df['Cluster'] = prediction
print(df)

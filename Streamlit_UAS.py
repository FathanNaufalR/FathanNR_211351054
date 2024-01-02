import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

df = pd.read_csv('day.csv')

df2 = df.drop(['dteday','season','yr','instant','mnth','holiday','weekday', 'workingday','weathersit','temp','atemp','hum','windspeed'], axis=1)

st.header("Isi Dataset")
st.write(df)


# elbow

x1 = df2["registered"]
x2 = df2["casual"]
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)

for k in K:
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)

    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)

    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / X.shape[0]
    mapping2[k] = kmeanModel.inertia_

for key, val in mapping1.items():
  st.write(f'{key} : {val}')

plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()

st.set_option('deprecation.showPyplotGlobalUse', False) 
elbo_plot = st.pyplot()

st.sidebar.subheader("Nilai Jumlah K")
clust = st.sidebar.slider("Pilih jumlah cluster :", 2,10,3,1)

def k_means(n_clust):
  kmeans = KMeans(n_clusters=n_clust).fit(X)
  centroids = kmeans.cluster_centers_
  st.write(centroids)
  df3={'x':df["registered"],'y':df["casual"]}
  fig,ax=plt.subplots()
  plt.scatter(df3['x'], df3['y'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
  plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
  plt.xlim([-500, 9000])
  plt.ylim([-500, 4000])
  ax.legend()
  st.header('Cluster Plot')
  st.pyplot()
  st.write(df2)
  
k_means(clust)
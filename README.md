# Laporan Tugas Machine Learning
## Nama : Fathan Naufal Rosidin
## NIM : 211351054
## Kelas : Informatika Pagi B

## Domain Proyek
  Mengelompokan data penyewa sepedah, yaitu data penyewa casual dan penyewa yang sudah registerasi

## Business Understanding
Sistem berbagi sepeda generasi baru dari persewaan sepeda di mana seluruh proses mulai dari keanggotaan, penyewaan, dan pengembalian menjadi otomatis.

### Problem Statement
- Penyewa harus datang ke tempat penyewaan sepedah, lalu mengurus registerasi, baru bisa menggunakan sepedah yang dimana membutuhkan banyak proses
  
### Goals
- Penyewa dapat dengan mudah menyewa sepeda dari posisi tertentu dan kembali lagi ke posisi lain dimana si penyewa tinggal datang dan memakai sepedahnya.

### Solution Statement
- Membuat penelitian pengelompokan data penyewa sepedah casual dan penyewa register (dengan algoritma K-MEAN)


## Data Understanding
https://www.kaggle.com/datasets/lakshmi25npathi/bike-sharing-dataset


## Import Dataset
dalam pemanggilan data, pertama-tama saya mengupload file kaggle.json agar bisa mendapatkan akses pada kaggle
```python
from google.colab import files
files.upload()
```
setelah kode di run lalu upload file kaggle.json yang kita dapat dari kaggle


Setelah itu saya membuat direktori dan izin akses pada skrip ini
```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```

Lalu mendownload Dataset yang sudah di pilih
```python
!kaggle datasets download -d lakshmi25npathi/bike-sharing-dataset
```

Karena dataset yang terdownload berbentuk ZIP, maka kita Unzip terlebih dahulu datasetnya
```python
!mkdir bike-sharing-dataset
!unzip bike-sharing-dataset.zip -d bike-sharing-dataset
!ls bike-sharing-dataset
```

## Import Library
 Mengimport library yang di butuhkan
 
 ```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
```

## Data Discovery

### penjelasan variabel pada Bike Sharing dataset :
- instant: Record Index (int64)
- dteday : Tanggal (object)
- season : Musim (int64)
- yr : Tahun (int64)
- mnth : Bulan (int64)
- holiday : Hari libur (int64)    
- weekday : Hari pada 1 minggu (int64)  
- workingday : Hari kerja (int64)
- weathersit : Cuaca (int64)
- temp : Suhu normal dalam Celcius. (float64)
- atemp: Suhu perasaan dinormalisasi dalam Celsius. (float64)      
- hum: Kelembaban yang dinormalisasi. (float64)
- windspeed: Kecepatan angin dinormalisasi. (float64)
- casual: jumlah pengguna biasa (int64)
- registered: jumlah pengguna terdaftar (int64)                            
- cnt: hitungan total sepeda sewaan termasuk sepeda kasual dan terdaftar (int64)  


Merubah nama pemanggilan data menjadi df agar mudah untuk di panggil
```python
df = pd.read_csv('/content/bike-sharing-dataset/day.csv')
```

Memunculkan data pada dataset dengan default 5 baris
```python
df.head()
```

Mengetahui deskripsi pada data seperti tipedata
```python
df.info()
```

Melihat ada berapa jumlah baris dan kolom
```python
df.shape
```

Memeriksa apakah ada nilai null pada dataset
```python
df.isna().sum()
```

Memeriksa apakah ada data yang duplikat pada dataset
```python
df.duplicated().sum()
```

Merubah nilai numerik pada dataset menggunakan Map() , untuk mempermudah memahami plot pada analisis kita.
```python
df['season']=df.season.map({1: 'spring', 2: 'summer',3:'fall', 4:'winter'})
df['mnth']=df.mnth.map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'June',7:'July',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})
df['weathersit']=df.weathersit.map({1: 'Clear',2:'Mist + Cloudy',3:'Light Snow'})
df['weekday']=df.weekday.map({0:'Sun',1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat'})
df.head()
```

## EDA


Mengecek korelasi antar variabel
```python
plt.figure(figsize=(20, 12))
sns.heatmap(df.corr(), annot = True)
plt.title("Korelasi antar variabel")
plt.show()
```
![image](https://github.com/FathanNaufalR/FathanNR_211351054/assets/149129682/c4041355-10f2-4685-9a26-52fc06676b7b)



Histogram
```python
columns = ['casual', 'registered', 'cnt']

fig, ax = plt.subplots(1, 3, figsize=(10,5))

for i, ax in enumerate(ax):
    sns.histplot(x=df[columns[i]], ax=ax, bins=10, color='red')
    ax.set_title(columns[i])
    ax.set_xlabel("")
    ax.set_ylabel("")

plt.tight_layout()
plt.show()
```
![image](https://github.com/FathanNaufalR/FathanNR_211351054/assets/149129682/49c04c50-7e57-48b0-af4f-0035521f7841)



Memahami data menggunakan boxplot
```python
plt.figure(figsize=(22, 10))

plt.subplot(2,3,1)
sns.boxplot(x = 'yr', y = 'cnt', data = df,palette="icefire")

plt.subplot(2,3,2)
sns.boxplot(x = 'holiday', y = 'cnt', data = df,palette="vlag")

plt.subplot(2,3,3)
sns.boxplot(x = 'workingday', y = 'cnt', data = df,palette="Spectral")

plt.subplot(2,3,4)
sns.boxplot(x = 'mnth', y = 'cnt', data = df,palette="coolwarm")


plt.subplot(2,3,5)
sns.boxplot(x = 'weekday', y = 'cnt', data = df,palette="rocket")

plt.show()
```
![image](https://github.com/FathanNaufalR/FathanNR_211351054/assets/149129682/ab45f867-3684-4809-a309-1d4f6ee10289)



Memeriksa outlier pada data
```python
sns.boxplot(df)
fig=plt.gcf()
fig.set_size_inches(15,10)
```
![image](https://github.com/FathanNaufalR/FathanNR_211351054/assets/149129682/8b46c949-c1ae-46ea-a196-af825d320a10)



Melihat data sepeda yang di sewa berdasarkan musim menggunakan barplot
```python
su=df.loc[df['season'] == 'summer', 'cnt'].sum()
sp=df.loc[df['season'] == 'spring', 'cnt'].sum()
fa=df.loc[df['season'] == 'fall', 'cnt'].sum()
wi=df.loc[df['season'] == 'winter', 'cnt'].sum()

data = {
  "season": ['summer', 'spring', 'fall', 'winter'],
  "cnt": [su, sp, fa,wi]
}

df0 = pd.DataFrame(data)
```
```python
plt.bar(x='season',height='cnt',data=df0,color='darkblue')
plt.show()
print(df0)
```
![image](https://github.com/FathanNaufalR/FathanNR_211351054/assets/149129682/649b4813-9c7c-4ff0-b224-0d563f726d72)


## Data Preprocessing

Karena dilihat dari boxplotnya kolom casual memiliki outlier yang banyak, maka akan dilakukan drop data
```python
Q1 = (df['casual']).quantile(0.25)
Q3 = (df['casual']).quantile(0.75)
IQR = Q3 - Q1

maximum = Q3 + (1.5*IQR)
minimum = Q1 - (1.5*IQR)

kondisi_lower_than = df['casual'] < minimum
kondisi_more_than = df['casual'] > maximum

df.drop(df[kondisi_lower_than].index, inplace=True)
df.drop(df[kondisi_more_than].index, inplace=True)
```

karena saya akan mengcluster kolom casual, dan registered saja maka akan dilakukan drop data
```python
df2 = df.drop(['dteday','season','yr','instant','mnth','holiday','weekday', 'workingday','weathersit','temp','atemp','hum','windspeed'], axis=1)
df2.head()
```

## Modeling


Melakukan clustering pada Kolom Registered dan Casual, dengan menggunakan K-Means
```python
x1 = df["registered"]
x2 = df["casual"]
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

# Visualizing the data
plt.plot()
plt.xlim([-500, 9000])
plt.ylim([-500, 4000])
plt.title('Dataset Registered dan Casual')
plt.scatter(x1, x2)
plt.show()
```
![image](https://github.com/FathanNaufalR/FathanNR_211351054/assets/149129682/7591716f-8fd6-44fa-bf59-0bb62d5e8c3a)


#lalu saya melakukan elbow
```python
distortions = []
inertias = []
mapping1 = {}
mapping2 = {}
K = range(1, 10)

for k in K:
    #Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)

    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                        'euclidean'), axis=1)) / X.shape[0])
    inertias.append(kmeanModel.inertia_)

    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_,
                                   'euclidean'), axis=1)) / X.shape[0]
    mapping2[k] = kmeanModel.inertia_
```
```python
for key, val in mapping1.items():
    print(f'{key} : {val}')
```
```python
plt.plot(K, distortions, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('Distortion')
plt.title('The Elbow Method using Distortion')
plt.show()
```
![image](https://github.com/FathanNaufalR/FathanNR_211351054/assets/149129682/c093e2a1-5845-4ba6-962c-77f4f11d292c)

Dari elbow method , ditemukan bahwa jumlah kluster yang dapat di gunakan adalah 3. Tapi dikarnakan elbow method terkadang ambigu , supaya yakin maka perlu kita lakukan silhouette method


```python
# Standarisasi (Opsional tapi terkadang di sarankan)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df2)

# Mengurangi dimensi tinggi dengan Principal Component Analysis (PCA) (Opsional tapi terkadang dapat membantu)
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Untuk set jumlah cluster
cluster_range = range(2, 11)

# untuk menyimpan skor kluster pada setiap jumlah kluster
silhouette_scores = []

for n_clusters in cluster_range:
    # untuk Fit model K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(df_pca)  # Untuk mengurangi dimensi tinggi data kita

    # Kalkulasi skor silhouette
    silhouette_avg = silhouette_score(df_pca, cluster_labels)
    silhouette_scores.append(silhouette_avg)

plt.plot(cluster_range, silhouette_scores, marker='o')
plt.title('Silhouette Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()
```
![image](https://github.com/FathanNaufalR/FathanNR_211351054/assets/149129682/757da7e9-d237-4c6c-ac0c-3ccb49462e76)

Ditemukan skor silhouette tertinggi adalah kisaran 0.48 pada jumlah kluster 4. Maka kita akan menggunakan Jumlah kluster 4 pada K-Means nya.


## Visualisasi hasil algoritma

```python
df3={'x':df["registered"],'y':df["casual"]}
plt.xlim([-500, 9000])
plt.ylim([-500, 4000])
plt.scatter(df3["x"], df3["y"])
plt.title('Dataset Registered dan Casual')
plt.show()
```
![image](https://github.com/FathanNaufalR/FathanNR_211351054/assets/149129682/e3de3c53-25ae-4ed8-b70c-d44037c262eb)



Melakukan K-mean clustering
```python
kmeans = KMeans(n_clusters=4).fit(X)
centroids = kmeans.cluster_centers_
print(centroids)

fig,ax=plt.subplots()
plt.scatter(df3['x'], df3['y'], c=kmeans.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.xlim([-500, 9000])
plt.ylim([-500, 4000])
ax.legend()
plt.show()
```
![image](https://github.com/FathanNaufalR/FathanNR_211351054/assets/149129682/8cc2304d-dd5f-46d5-b62f-5724a3f58c52)


## Save model (pickle)

```python
kmeans = KMeans(n_clusters=4).fit(X)

with open('kmeans_model.pkl', 'wb') as file:
    pickle.dump(kmeans, file)

with open('kmeans_model.pkl', 'rb') as file:
    loaded_kmeans = pickle.load(file)
```

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

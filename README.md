# Laporan Proyek Machine Learning - Anthony Kevin Oktavius

## Domain Proyek
### Latar Belakang

![Kecimen dan Besni](https://image.shutterstock.com/image-photo/black-raisins-on-white-background-260nw-1262583304.jpg)

Kismis adalah salah satu bagian dari buah. Buah ini berasal dari anggur yang dikeringkan. Banyak yang menjadikan buah ini menjadi snack, dekorasi pada makanan (misalnya roti), dan bahan untuk alkohol.

Salah satu aplikasi dari pengecekan ukuran kismis adalah pengecekan kualitas kismis. Metode pengecekan kualitas kismis masih secara manual. Namun, masalahnya, pengecekan kualitas kismis secara manual membutuhkan waktu yang lama dan membutuhkan ketelitian. Terlebih lagi, bisa saja terdapat kesalahan pada manusia terkait pengecekan kualitas kismis, sehingga kismis yang tidak berkualitas baik terdapat pada kemasan. Maka dari itu, perlu adanya penerapan machine learning yang bisa mempercepat pengecekan kualitas kismis. 

Ide di atas membuat saya ingin menggunakan machine learning. Zaman sekarang, ada banyak algoritma machine learning. Menurut saya, masalah ukuran kismis bisa diukur dengan menggunakan machine learning berupa model linear. Hal ini karena fitur-fitur seperti diameter, area, dan panjang saling berkaitan satu sama lain. 

Uraian di atas menunjukkan motivasi saya untuk membangun suatu proyek klasifikasi kismis. Proyek ini adalah masalah klasifikasi. Klasifikasi adalah kegiatan pengelompokkan suatu data, sehingga bisa dikategorikan ke dalam label-label yang diberikan. Data label-label tersebut disajikan dalam bentuk kategorikal. Data kategorikal ini adalah data nominal, karena label pada kismis tidak menunjukkan tingkatan apapun (hanya menunjukkan kualitas kismis tersebut).

### Sumber referensi
* ÇINAR, İ., KOKLU, M., & TAŞDEMİR, Ş. (2020). Classification of raisin grains using machine vision and artificial intelligence methods. Gazi Mühendislik Bilimleri Dergisi (GMBD), 6(3), 200-209.
* Tarakci, F., & Ozkan, I. A. (2021). Comparison of classification performance of kNN and WKNN algorithms. Selcuk University Journal of Engineering Sciences, 20(2), 32-37.

## Business Understanding
### Problem Statements
* Fitur apa yang paling berpengaruh pada ukuran kismis?
* Dari fitur yang ada, apakah kismis tersebut termasuk pada Kecimen atau Besni?

### Goals
* Mengetahui fitur yang paling berkorelasi dengan ukuran kismis.
* Membuat model machine learning untuk memprediksi kategori ukuran kismis dengan seakurat mungkin.

### Solution Statements
Seperti penjelasan singkat pada latar belakang, model yang akan digunakan adalah model linear. Alasannya adalah fitur-fitur yang digunakan untuk mengkategorikan ukuran kismis saling berkorelasi satu sama lain. Kita akan menggunakan tiga model machine learning, yaitu sebagai berikut.

* KNN
* Linear Classification dengan SVD
* SVC

Dalam proyek ini, masalah yang diajukan adalah masalah klasifikasi. Maka dari itu, berikut adalah metrik-metrik yang digunakan.

* Confusion Matrix
* Precision
* Recall
* F1-Score
* Accuracy

## Data Understanding

**Deskripsi Data**:
Data kismis yang telah diukur dengan menggunakan CVS. Data-data kismis ini berasal dari Turkey.  Data-data ini juga telah dilakukan preprocess, sehingga saya tidak perlu melakukan terlalu banyak preprocessing.

**Data dapat diunduh pada link berikut**. https://archive.ics.uci.edu/ml/machine-learning-databases/00617/Raisin_Dataset.zip

**Deskripsi fitur-fitur pada data dapat dilihat sebagai berikut.**
* Area: jumlah pixel yang membentuk buah kismis. (float, range: 25387 - 235047)
* Perimeter: keliling kismis (float, range: 619 - 2697)
* MajorAxisLength: ukuran terpanjang pada kismis tersebut. (float, range: 225 - 997)
* MinorAxisLength: ukuran terpendek pada kismis tersebut (float, range: 143 - 492)
* Eccentricity: tingkat bentuk kurva pada kismis (float, range: 0.348730 - 0.962124) 
* ConvexArea: luas cangkang terkecil pada kismis (float, range: 26139 - 278217)
* Extent: rasio suatu area yang dibentuk antara kulit luar terkecil dengan kotak yang dibentuk dari kismis (float, range: 0.379856 - 0.835455)
* Class: menentukan tipe kismis (kecimen atau besni)

**Jumlah data**: 900

**Jumlah label kecimen dan besni**: seimbang (masing-masing memiliki 450 data)

**Kondisi data**: clean, sehingga jumlah preprocessing pada dataset ini tidak terlalu banyak

**Langkah-langkah pemahaman data yang dilakukan adalah sebagai berikut.**
1. **Deskripsi Variabel**: Melihat deskripsi data dengan numpy dengan info() dan describe(). 
    * **info()**: menjelaskan jumlah kolom yang non-null dan data type. 
    * **describe()**: menjelaskan beberapa informasi mengenai statistika suatu data.

2. **Penanganan missing value**: 
    Berikut adalah tahapan penanganan missing value yang tidak perlu dilakukan:
    * Tidak ada data yang kosong.
    * Tidak ada value yang memiliki nilai 0 pada semua fitur-fitur numerikal.
    * Tidak ada ukuran MinorAxisLength yang lebih besar daripada MajorAxisLength
    
    Teknik penanganan missing value yang dilakukan:
    * **Penghapusan outlier:**
      Outlier adalah suatu data yang memiliki nilai yang berada jauh daripada rentang aslinya. Data ini sangat jarang kemunculannya, sehingga bisa memengaruhi nilai prediksi suatu data.
      
      ![Outlier](https://github.com/MaxZx3000/Exploratory-Data-Raisins/blob/main/submission-1-images/outlier.png?raw=true)
    
       Pada contoh di atas, ada satu data yang nilainya diatas nilai yang seharusnya (titik paling kanan). Maka dari itu, kita harus menghilangkan data tersebut. Salah satu teknik yang digunakan untuk menghapus outlier adalah teknik IDR (Interquartile Range). IDR menggunakan kuartil ketiga (Q3) dan kuartil pertama (Q1) untuk mendapatkan data-data yang berada di antara kuartil-kuartil tersebut. Berikut adalah rumus untuk menentukan nilai dibawah Q1 dan nilai diatas Q3: 
       ```
       Batas bawah = Q1 - 1.5 * IQR
       Batas atas = Q3 + 1.5 * IQR
       ```
       Apabila data-data ada di luar nilai tersebut, maka data-data tersebut akan dihapus.

3. **Univariate Analysis**: variasi univariate ini bisa digunakan dengan tahap-tahap berikut.

    * **Categorical Features**
    Tujuan dari pengukuran fitur-fitur kategorikal aalah agar saya bisa melihat keseimbangan data antar satu label dengan label lainnya.
    Berikut adalah contoh persebaran datanya.
    ![Categorical Features Histogram](https://github.com/MaxZx3000/Exploratory-Data-Raisins/blob/main/submission-1-images/categorical_features.png?raw=true)
    Dari grafik di atas, kita bisa melihat bahwa persebaran data untuk kecimen dan besni masih seimbang, meskipun dilakukan penghapusan outliers.
    
    * **Numerical Features**
    Di sini, saya melakukan analisis histogram untuk melihat persebaran data pada masing-masing numerik. Berikut adalah contoh data histogram yang saya peroleh.
    ![Hisrogram Numerical Features](https://github.com/MaxZx3000/Exploratory-Data-Raisins/blob/main/submission-1-images/numerical_features.png?raw=true)
    Dari grafik di atas, kita bisa memperoleh data sebagai berikut.
    * **Field Area, MajorAxisLength, MajorAxisLength, MinorAxisLength** memiliki bentuk kurva yang hampir sama. Ini berarti, ada peluang bahwa mereka memiliki korelasi yang tinggi.
    
    * **Field Extent dan field eccentricity** tidak memiliki hubungan relasi apapun, maupun field-field yang dijelaskan pada poin sebelumnya.

4. **Multivariate Analysis:**
    * **FacetGrid**
    Tujuan dari grafik ini adalah cara untuk mengetahui hubungan antara rata-rata nilai masing-masing fitur numerikal dengan fitur kategorikal.
    ![Contoh plotting facetgrid](https://github.com/MaxZx3000/Exploratory-Data-Raisins/blob/main/submission-1-images/facet_grid.png?raw=true)
    Pada contoh grafik di atas, kita bisa melihat bahwa rata-rata pada MajorAxisLength memiliki dampak kecil terhadap kelas kecimen dan besni. Meskipun demikian, mereka memberi informasi bahwa kismis Besni memiliki major axis length lebih besar daripada kismis kecimen.
    
    * **Pairplot**
    Pairplot ini bertujuan agar kita bisa melihat korelasi antara dua nilai. Plot ini memberi tahu kita mengenai korelasi antara fitur yang satu dengan fitur yang lainnya. Dengan demikian, kita bisa mereduksi fitur-fitur tersebut dengan teknik dimensionality reduction. Berikut adalah contoh hasil dari pairplot.
    ![Visualisasi Pairplot](https://github.com/MaxZx3000/Exploratory-Data-Raisins/blob/main/submission-1-images/pairplot.png?raw=true)
    
    * **Feature Selection dengan algoritma ANOVA**: 
    ANOVA adalah salah satu algoritma untuk melakukan analisis variance. Variance adalah nilai informasi yang terdapat pada atribut yang ada. Tujuan dari analisis variance ini adalah agar rata-rata dari dua atau lebih sampel data berasal dari distribusi yang sama. ANOVA menggunakan kalkulasi F-Statistic, yaitu suatu tes statistika yang digunakan untuk menghitung kalkulasi rasio antar variance. 
        
        ANOVA ini digunakan ketika ada satu variabel numerik dan satu variabel kategorikal. Hasil akhir dari tes ini digunakan untuk menghapus fitur-fitur yang tidak berkorelasi antar satu sama lain. Untuk melakukan pengecekan terhadap prioritas-prioritas fitur-fitur yang ada, kita bisa menggunakan library Sklearn SelectKBest untuk mengecek skor-skor dari masing-masing fitur.
        ![Skor kepentingan fitur dari ANOVA](https://github.com/MaxZx3000/Exploratory-Data-Raisins/blob/main/submission-1-images/anova.png?raw=true)
    
        Jika kita menggunakan sklearn, kita bisa melakukan output skoring kepentingan masing-masing fitur terhadap data kategorikal. Pada contoh di atas, kita bisa melihat bahwa fitur 3 dan 6 dapat dihapus, karena memiliki skor kepentingan yang sangat kecil.

* **Feature Selection dengan pearson correlation matrix:**
Ia menunjukkan relasi kekuatan linear antara dua variabel atau lebih. Correlation matrix ini berfungsi agar kita bisa melihat korelasi fitur berupa numerik. Nilai dari korelasi ini dapat dijabarkan sebagai berikut.
    * Jika nilai korelasi mendekati **angka -1**, maka korelasi fitur tersebut **kuat** 
    * Jika nilai korelasi mendekati **angka 0**, maka **tidak ada korelasi** antar fitur-fitur tersebut
    * Jika nilai korelasi mendekati **angka 1**, maka korelasi fitur tersebut **kuat, namun memiliki makna kebalikan**
    ![Contoh Correlation Matrix](https://github.com/MaxZx3000/Exploratory-Data-Raisins/blob/main/submission-1-images/correlation%20matrix_2.png?raw=true)

    Apabila kita coba lihat pada correlation matrix di atas, kita bisa menggabungkan beberapa fitur dengan penjelasan seperti berikut.
    * ConvexArea dan Area, karena skor korelasinya adalah 1
    * MajorAxisLength dan Perimeter, karena skor korelasinya adalah 0.98

## Data Preparation
Dalam proyek saya, saya menerapkan beberapa tahapan persiapan data, yaitu sebagai berikut.

* **Reduksi dimensi dengan menggunakan PCA:**
PCA adalah singkatan dari Principal Component Analysis. Principal component analysis bertujuan untuk mentransformasikan data dengan mengubah dimensi fitur-fitur yang ada menjadi dimensi yang lebih kecil lagi. Ia mereduksi dimensi dengan memiliki variance maksimal, yaitu rata-rata jarak kuadrat dari poin-poin yang telah diproyeksi dengan garis lurus ke titik asal. Inilah yang dimaksud dengan Principal Components. Saat teknik ini diterapkan, hanya komponen utama yang akan digunakan. Inilah yang dimaksud dengan Principal Component (PC). Jika data-data pada fitur berkorelasi linear, maka kita hanya menggunakan PC pertama saja. PC kedua dan seterusnya merupakan sisa informasi-informasi yang tidak didapatkan pada data pertama. Dalam sklearn. kita bisa menggunakan atribut score untuk melihat nilai PCA.

* **Splitting training dan testing set:**
Hal ini bertujuan agar kita bisa mengukur accuracy dengan data yang belum pernah dipelajari. Ukuran training dan testing dapat ditentukan bebas. Jika dilihat pada dataset setelah penghapusan outlier, datasetnya terdapat 795 data. Maka dari itu, kita bisa mengambil test data sebesar 20% saja, karena kita masih memiliki 636 data untuk training.

* **Konversi label menggunakan LabelEncoder:**
Tujuan dari label encoder ini adalah agar kita bisa melakukan konversi kata-kata kategorikal menjadi numerikal. Machine Learning hanya bisa menggunakan angka sebagai pemrosesan data. Lalu, ia juga merepresentasikan hubungan prioritas/ tingkatan. Label-label di dataset memiliki hubungan bahwa Besni memiliki ukuran yang lebih kecil daripada kecimen. Maka dari itu, kita bisa menggunakan LabelEncoder.

* **Standarisasi label menggunakan StandardScaler:** 
StandardScaler bertujuan agar machine learning mampu menganalisis data dengan lebih cepat, terutama ketika machine learning bekerja dengan melakukan perhitungan batching dan jarak. Ia mengubah rata-rata pada suatu data menjadi 1 dan standar deviasi menjadi 0. Karena fitur-fitur numerikal kita mendekati standar distribusi normal, ia akan tetap mempertahankan informasi esensial data-data tersebut, sehingga StandardScaler aman untuk diterapkan.

## Modeling
Ada tiga model yang saya gunakan untuk projek ini:
* **KNN**:
    Parameter yang digunakan:
    * **n_neighbors**: jumlah neighbour yang ingin digunakan
    * **p**: parameter power untuk metrik minkowski. 
        * Jika p = 1, maka menggunakan jarak manhattan (l1)
        * Jika p = 2, maka menggunakan jarak eucledian (l2)
        * Selain angka di atas, maka ia akan menggunakan minkowski distance (l_p)
    * **leaf_size**: ukuran pohon, jika menggunakan parameter algoritma BallTree dan KDTree. Hal ini akan mempengaruhi kecepatan konstruksi algoritma dan memori yang dibutuhkan.

* **Linear Classification dengan SGD**:
    Parameter yang digunakan:
    * **max_iter**: angka iterasi maksimum yang ingin dijalankan 
    * **tol**: angka toleransi agar algoritma berhenti. Ini terjadi apabila angka loss antara iterasi yang sedang dijalankan dan iterasi sebelumnya tidak terlalu berbeda jauh, sehingga bisa dihentikan iterasi algoritma ini.
    * **random_state**: random number generator yang berguna untuk mengacak data. Jika kita menyertakan angka random statenya, kita akan mendapatkan angka yang sama setiap kali kita jalankan algoritmanya, sehingga cocok untuk melakukan debugging algoritma.

* **Linear SVC**
    Parameter yang digunakan:
    * **max_iter**: angka iterasi maksimum yang ingin dijalankan 
    * **tol**: angka toleransi agar algoritma berhenti. Ini terjadi apabila angka loss antara iterasi yang sedang dijalankan dan iterasi sebelumnya tidak terlalu berbeda jauh, sehingga bisa dihentikan iterasi algoritma ini.
    * **random_state**: random number generator yang berguna untuk mengacak data. Jika kita menyertakan angka random statenya, kita akan mendapatkan angka yang sama setiap kali kita jalankan algoritmanya, sehingga cocok untuk melakukan debugging algoritma.

Berikut adalah kelebihan dan kekurangan masing-masing algoritma:

* **KNN**
    
    * **Kelebihan:** 
    
      * Mudah untuk diimplementasikan dan dimengerti
      * Hanya membutuhkan satu parameter saja, yaitu jumlah tetangga (K)
      * Banyak pilihan metrik jarak, seperti Eucledian, Minkowski, Manhattan, dan lain-lain.
      * Model ini merupakan model non-linear, sehingga cocok untuk diterapkan bila data kita tidak bisa dilatih dengan model yang memprediksi dengan cara linear.
        
    * **Kekurangan:** 

      * Performa lambat jika menggunakan data yang banyak. KNN melihat seluruh dataset yang ada. Time complexitynya memiiki nilai O(MN log (k)). Hal ini juga berlaku untuk dimensi yang tinggi.
      * KNN menanggap bahwa semua fitur itu berjarak sangat dekat. Salah satu pendekatan yang bisa dilakukan adalah melakukan scaling pada setiap data yang ada.
      * Rentan terhadap outlier. Hal ini akan lebih berdampak jika kita berhadapan dengan dimensi yang tinggi. Dimensi yang tinggi ini membuat rata-rata jarak pemisahan lebih besar.

* **Linear Classification dengan SGD**
    
    * **Kelebihan:**
      
      * Lebih cepat, karena hanya satu sampel training yang diproses pada satu waktu.
      * Performa konvergensi lebih cepat.
    
    * **Kelemahan:**
        
      * Pada saat training, jika datanya bersifat noisy, maka gradient descent bisa saja akan menjauh dari titik optima.
      * Tidak ramah untuk memori, karena kita harus memproses semua data training.
        
* **Linear SVM**
    
    * **Kelebihan:**
      * Cocok untuk dataset yang tidak memiliki outlier (tidak ada noise).
      * Bisa diterapkan untuk dimensi yang tinggi.
    
    * **Kelemahan:**
      
      * Peforma rendah jika berurusan dengan dataset yang banyak
      * Kurang bisa dalam menentukan garis yang tepat untuk dataset dengan kelas yang saling bertabrakkan satu dengan yang lain.

Berikut adalah **hasil training** dari tiga model tersebut (hasil ini sudah diperoleh dengan manual dan grid search parameter tuning):

```
Accuracy Score

KNN: 0.8867924528301887
Linear Classification with SGD: 0.8805031446540881
Linear SVM: 0.8867924528301887
```

Apabila kita perhatikan, perbedaan accuracy antara model yang satu dengan model lainnya tidak terlalu berbeda jauh, jika dicoba beberapa kali. Untuk melihat penyebabnya, berikut adalah visualisasi datanya.

![Visualisasi Data setelah PCA](https://github.com/MaxZx3000/Exploratory-Data-Raisins/blob/main/submission-1-images/pairplot%202.png?raw=true)

Dari grafik di atas, dengan menggunakan model linear classification sederhana, kita bisa melihat bahwa model tersebut sudah memiliki lokasi dan arah garis pemisahan yang cukup baik, meskipun ada sedikit bagian data yang overlapping. Setelah saya mencoba untuk menggunakan model yang lebih kompleks, seperti KNN dan Linear SVM, saya mendapatkan bahwa accuracy tidak terlalu jauh dengan model linear sederhana. Maka dari itu, dengan menggunakan linear SGD saja, kita sudah bisa mendapatkan nilai yang baik tanpa memerlukan model yang kompleks.

## Evaluation

Sebelum kita menghitung metrik-metrik yang dibutuhkan untuk klasifikasi, kita bisa menggunakan komponen confusion matrix. Berikut adalah contoh gambar confusion matrix.

![Contoh Confusion Matrix](https://github.com/MaxZx3000/Exploratory-Data-Raisins/blob/main/submission-1-images/confusion_matrix.png?raw=true)

Dari gambar confusion matrix di atas, komponen-komponen dasar pembentuk confusion matrix sebagai berikut.

* **True Positive**: jumlah data berlabel **positif** yang diprediksi **benar**.
* **True Negative**: jumlah data berlabel **positif** yang diprediksi **salah**.
* **False Positive**: jumlah data berlabel **negatif** yang diprediksi **benar**.
* **False Negative**: jumlah data berlabel **negatif** yang diprediksi **benar**.

Dari penjelasan confusion matrix di atas, kita bisa menghitung metrik-metrik yang dibutuhkan untuk klasifikasi. Berikut adalah evaluasi metrik yang digunakan pada proyek ini.

* **Precision**: nilai yang menunjukkan kemampuan machine learning untuk melakukan prediksi berlabel positif secara tepat.
Rumus precision adalah sebagai berikut.

    ![Precision Formula](https://miro.medium.com/max/888/1*C3ctNdO0mde9fa1PFsCVqA.png)

* **Recall**: nilai yang menunjukan jumlah data berlabel positif yang dapat dideteksi, Rumus recall adalah sebagai berikut.
  
    ![Recall Formula](https://miro.medium.com/max/836/1*dXkDleGhA-jjZmZ1BlYKXg.png)

* **F1-Score**: nilai yang menunjukkan jumlah prediksi benar yang dapat dilakukan. Rumus F1-Score adalah sebagai berikut.

    ![F1 Score Formula](https://www.gstatic.com/education/formulas2/397133473/en/f1_score.svg)

* **Support**: jumlah sampel yang benar pada setiap label yang terdapat pada suatu dataset.

* **Accuracy**: nilai pecahan yang menunjukkan rasio antara jumlah data yang diprediksi benar dengan total jumlah data yang ada.

Apabila kita coba cek dari beberapa pengulangan training pada model pada projek ini, kurang lebih, kita dapat mendapatkan hasil classification matrix sebagai berikut.
```
KNN
              precision    recall  f1-score   support

     Kecimen       0.87      0.89      0.88        73
       Besni       0.90      0.88      0.89        86

    accuracy                           0.89       159
   macro avg       0.89      0.89      0.89       159
weighted avg       0.89      0.89      0.89       159

Linear Classification with SGD
              precision    recall  f1-score   support

     Kecimen       0.88      0.87      0.87        76
       Besni       0.88      0.89      0.89        83

    accuracy                           0.88       159
   macro avg       0.88      0.88      0.88       159
weighted avg       0.88      0.88      0.88       159

Linear SVM
              precision    recall  f1-score   support

     Kecimen       0.88      0.88      0.88        75
       Besni       0.89      0.89      0.89        84

    accuracy                           0.89       159
   macro avg       0.89      0.89      0.89       159
weighted avg       0.89      0.89      0.89       159
```

Dari hasil di atas, kita dapat menyimpulkan hasil sebagai berikut.
* Ketiga model di atas dapat melakukan prediksi untuk kecimen dan besni dengan baik.

* Nilai support pada kecimen lebih kecil daripada besni. Hal ini bisa saja terjadi karena ada sedikit ketidakseimbangan pembagian data pada modul 'train test split'. Maka dari itu, proporsi pembelajaran pada besni lebih besar daripada kesimen, sehingga mempengaruhi nilai support.

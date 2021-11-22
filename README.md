# Laporan Proyek Machine Learning - Jody Irawan

## Domain Proyek
Domain proyek yang Saya pilih untuk Ujian proyek machine learning Terapan ini yaitu Ekonomi dan bisnis tentang Prediksi Harga Mobil Audi.

### Latar Belakang
Audi adalah salah satu produsen mobil yang cukup terkenal di Eropa dan di seluruh dunia. Produsen mobil asal Jerman ini memasarkan berbagai model mobil yang mencakup beragam kelas dan segmen untuk kalangan menengah ke atas[[1](https://www.autofun.co.id/mobil/audi)]. Semakin hari, harga mobil Audi mengalami kenaikan harga karna fitur dan mesin mobil dari tahun ke tahun semakin baik. Produsen mobil pun mencari cara menetapkan harga yang pas untuk mobil baru yang akan keluar. Dengan Machine Learning ini, akan sangat membantu produsen mobil untuk menyelesaikan masalah ini. untuk itu saya akan mencoba memprediksi kasus ini dengan machine learning Model predictive alytics untuk memprediksi harga optimal berdasarkan data atau catatan historis penjualan juga berdasarkan teori dan praktik yang sudah dipelajari di kelas ini.

## Business Understanding

### Problem Statements
1. Berapa harga pasar mobil audi dengan karakteristik atau fitur tertentu?
2. Model algoritma apa yang cocok dan bagaimana kinerja algoritma untuk prediksi harga mobil Audi?

### Goals
1. Membuat model machine learning yang dapat memprediksi harga mobil Audi seakurat mungkin berdasarkan fitur-fitur yang ada.
2. menentukan Model terbaik untuk melakukan prediksi harga mobil audi.

### Solution statements
Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini diantaranya :
1.  Menangani Missing Value nilai bernilai 0 dengan cara melakukan drop terhadap data yang hilang pada kolom tax dan engineSize. Karena merunut saya nilai tax dan engine size tidak mungkin bernilai 0.
2. Mengatasi outliers dengan metode IQR.
3. melakukan drop pada kolom tax karena pada saat mengamati hubungan antara fitur numerik denagn fungsi pairplot() dan corr(), data yang ditampilkan kosong.
4. Membagi dataset menjadi data latih (train) dan data uji (test) dengan rasio 90% untuk data latih dan 10% untuk data uji.
5. Melakukan standarisasi dengan menerapkan StandardScaler pada data.
6. Melakukan pengembangkan model machine learning dengan tiga algoritma yaitu K-Nearest Neighbor, Random Forest, dan Boosting Algorithm. Setelah itu menentukan algoritma mana yang memberikan hasil prediksi terbaik. Untuk dataset, fitur dalam proses pembangun model yaitu model, year, transmission, mileage, fuelType, tax, mpg, engineSize. Lalu untuk  fitur targetnya yaitu price. Untuk pengertian dari algoritma yang digunakan adalah sebagai berikut :
    -   K-Nearest Neighbor, Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k-tetangga terdekat[[2](https://www.dicoding.com/academies/319/tutorials/18580)].
    -   Random forest, merupakan kombinasi dari masing – masing pohon (tree) dari model Decision Tree yang baik, dan kemudian dikombinasikan ke dalam satu model[[3](http://learningbox.coffeecup.com/05_2_randomforest.html)].
    -   Algoritma boosting, Yaitu algoritma yang bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan[[4](https://www.dicoding.com/academies/319/tutorials/18590)].
        
        

## Data Understanding
1. Data yang saya gunakan untuk proyek ini adalah dataset prediksi harga mobil Audi.
[Audi Car Price Prediction](https://www.kaggle.com/rohitagrawal362/audi-car-price-prediction).berisi informasi metriks dari mobil audi untuk 10667 jenis-jenis mobil yang berbeda. Terdapat 2 buah data numerik (tipe data float64), 4 buah data numerik (tipe data int64) dan 3 buah data kategori (tipe data object). Terdapat juga beberapa kolom yang memiliki data kosong diantaranya pada kolom tax dan engineSize.

2. Variabel-variabel pada Audi Car Price Prediction adalah sebagai berikut:
- model : merupakan jenis model mobil Audi.
- year : merupakan tahun di release mobil.
- price : harga mobil.
- transmission : merupakan jenis transmission yang dipakai di mobil Audi.
- mileage : jarak tempuh mobil.
- fuelType : jenis bensin.
- tax : pajak mobil.
- mpg : Mil per galon (mpg - A.S.), pemakaian bahan bakar.
- engineSize : Kapasitas mesin.

Setelah mengevaluasi skor korelasinya, gunakan fungsi corr() dapat diamati pada gambar bahwa fitur ‘tax’ memiliki nilai 0 atau tidak memiliki korelasi dengan fitur targer. Untuk itu kita bisa drop pada kolom 'tax'. Lalu, fitur ‘mpg’ dan ‘year’ memiliki skor korelasi paling besar dengan fitur target ‘price’. Artinya, fitur 'price' cukup berkorelasi tinggi dengan kedua fitur tersebut.
![CorrelationMatrix](https://github.com/jodyirawan/Proyek-Predictive-Analytics-Audi/blob/main/Data%20Gambar/CorrelationMatrix.png?raw=true)

## Data Preparation
1. Menangani Missing Value nilai bernilai 0.
Dikarnakan nilai pada fitur tax dan engineSize tidak mungkin kosong, patut diduga bahwa ini merupakan data yang tidak valid atau sering disebut missing value. Untuk itu disini saya menanganni missing value dengan teknik drop, dimana teknik ini menghapus atau melakukan drop terhadap data yang hilang. Alasan saya menggunakan teknik drop dikarenakan jumlah sampel yang cukup banyak yaitu 10667, sementara data dengan mising value sebanyak 536 sampel missing value. Disini saya masih memiliki memiliki 10131 sampel lainnya.
2. Mengatasi outliers dengan metode IQR.
Outliers adalah sampel yang nilainya sangat jauh dari cakupan umum data utama. Ia adalah hasil pengamatan yang kemunculannya sangat jarang dan berbeda dari data hasil pengamatan lainnya. Pada kasus ini, untuk mengatasi outliers saya menggunakan metode IQR. IQR adalah singkatan dari Inter Quartile Range. IQR memberi tahu kita rentang di mana sebagian besar nilai berada. Jangkauan antar kuartil dihitung dengan mengurangkan kuartil pertama dari kuartil ketiga. 
IQR = Q3 - Q1 
3. melakukan drop pada kolom tax.
Dikarenakan pada saat mengamati hubungan antara fitur numerik denagn fungsi pairplot() dan corr(), data yang ditampilkan kosong atau tidak memiliki korelasi maka kita bisa melakukan drop pada kolom tax.
4. Encoding Fitur Kategori dengan one-hot-encoding
Melakukan proses encoding fitur kategori dengan teknik one-hot-encoding agar variabel kategori berubah menjadi variabel numerik. Pasalnya, banyak persamaan machine learning yang hanya menerima nilai numerik, bukan nilai kategorik.
5. pembagian data latih dan uji dengan 90:10
Train-Test-Split, Agar dapat menguji performa model pada data sebenarnya, maka perlu dilakukan pembagian dataset kedalam dua bagian. pembagian data latih dan uji dengan 90:10 dikarenakan samapai disini sampel yang cukup banyak yaitu 4537 sampel, maka cukup dengan rasio 90:10, sampel test memiliki total 454 sudah cukup untuk pengujian. Tujuan dari data uji adalah untuk untuk mengukur kinerja model pada data baru. Pembagian dataset dilakukan dengan modul train_test_split.
6. Melakukan standardisasi dengan menerapkan StandardScaler pada data.
Standarisasi menggunakan teknik StandarScaler dari library Scikitlearn. StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi. Proses standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma.

## Modeling
Pada tahap ini, saya akan mengembangkan model machine learning dengan tiga algoritma. yaitu :
1. K-Nearest Neighbor. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k-tetangga terdekat. Pemilihan nilai k pada KNN sangat penting dan berpengaruh terhadap performa model. Jika kita memilih k yang terlalu rendah, maka akan menghasilkan model yang overfit dan hasil prediksinya memiliki varians tinggi. Jika kita memilih k terlalu tinggi, maka model yang dihasilkan akan underfit dan prediksinya memiliki bias yang tinggi. Oleh karena itu pada kasus ini saya menggunakan k = 10 karena cukup ideal dan hasil prediksi akan lebih smooth.
2. Random Forest. algoritma ini disebut sebagai random forest karena algoritma ini disusun dari banyak algoritma pohon (decision tree) yang pembagian data dan fiturnya dipilih secara acak. Di sini saya menerapkan algoritma ini pada dataset menggunakan library scikit-learn.
3. Boosting Algorithm. Algoritma boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan. 

setelah model selesai dilatih dan dievaluasi dengan 3 algoritma, yaitu KNN, Random Forest, dan Boosting, model Random Forest memberikan nilai eror yang paling kecil. Maka model inilah yang saya pilih sebagai model terbaik untuk melakukan prediksi harga mobil Audi.
![HasilLatih](https://github.com/jodyirawan/Proyek-Predictive-Analytics-Audi/blob/main/Data%20Gambar/HasilLatih.png?raw=true)

## Evaluation
- Metrik yang saya gunakan pada prediksi ini adalah MSE atau Mean Squared Error yang menghitung selisih rata-rata nilai sebenarnya dengan nilai prediksi.
![HasilLatih](https://github.com/jodyirawan/Proyek-Predictive-Analytics-Audi/blob/main/Data%20Gambar/MSE.png?raw=true)
- MSE digunakan untuk memeriksa seberapa dekat perkiraan atau prakiraan dengan nilai aktual. Semakin rendah MSE, semakin dekat perkiraan dengan aktual. Berikut hasilnya :
![HasilLatih](https://github.com/jodyirawan/Proyek-Predictive-Analytics-Audi/blob/main/Data%20Gambar/HasilLatih.png?raw=true)
- Untuk mengujinya, mari kita buat prediksi menggunakan beberapa harga dari data test. Hasil prediksi akhirnya adalah sebagai berikut:
![HasiPrediksi](https://github.com/jodyirawan/Proyek-Predictive-Analytics-Audi/blob/main/Data%20Gambar/hasilPrediksi.png?raw=true)
Terlihat bahwa prediksi dengan Random Forest (RF) memberikan hasil yang paling mendekati.

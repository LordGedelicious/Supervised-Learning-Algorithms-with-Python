# Supervised-Learning-Algorithms-with-Python

## Daftar Isi
- [Spesifikasi](#spesifikasi)
- [Penggunaan](#penggunaan)
- [Identitas](#identitas)

## Spesifikasi
- Implementasikan algoritma KNN, Logistic Regression, dan ID3.
- KNN dan Logistic Regression hanya perlu mengklasifikasikan data bertipe numerik.
- ID3 hanya perlu mengklasifikasikan data bertipe kategorikal.
- KNN dapat menerima masukan nilai k yang dipakai.
- Logistic Regression dapat menerima masukan berupa jumlah epoch dan learning rate pembelajaran.
- Algoritma dapat menerima masukan dataset dalam format csv dengan beberapa fitur dan binary label.
- Implementasi dibebaskan menggunakan bahasa apapun.

Jawablah pertanyaan berikut!
1. Jelaskan yang dimaksud dengan supervised learning dan cakupannya!
2. Jelaskan cara kerja algoritma yang telah diimplementasikan!
3. Bandingkan ketiga algoritma tersebut, lalu tuliskan kelebihan dan kelemahannya!
4. Jelaskan penerapan dari algoritma supervised di berbagai bidang (misalnya industri atau kesehatan)!


Isi direktori adalah sebagai berikut:
```
├── docs [berisikan answers.pdf yang berisi jawaban dari pertanyaan wajib]
├── src [berisikan answers.pdf yang berisi jawaban dari pertanyaan wajib]
    ├── normalizeCSV.py [gunakan untuk normalisasi label target menjadi 0/1]
    ├── id3.py [menyimpan algoritma ID3]
    ├── knn.py [menyimpan algoritma KNN]
    ├── logisticRegression.py [menyimpan algoritma Regresi Logistic]
    ├── createFakeData.py [digunakan untuk membuat dataset random kategorikal is_flue.csv]
    ├── main.py [gunakan untuk mengakses semua fungsi lain]
├── testcase [berisikan testcase untuk dipanggil ke dalam program]
    ├── haberman.csv [berisikan file Haberman's Survival Dataset, numerical dataset]
    ├── is_flu.csv [berisikan file gejala penyakit flu, categorical dataset]
README.md
```

## Penggunaan
**[IMPORTANT NOTES]** 
1. Untuk metode logsitc regression, diwajibkan untuk melakukan normalisasi label target menjadi 0/1 untuk menghindari terjadinya kasus cross entropy error
2. Untuk metode ID3, diharapkan untuk tidak menggunakan dataset yang salah satu atributnya memiliki lebih dari dua atribut unik (seperti true/false, A/B, dll)

**[CARA PEMAKAIAN PROGRAM]**
1. Buka direktori `src`
2. Buka terminal [PASTIKAN PYTHON MINIMAL VERSI 3.9 SUDAH TERINSTAL]
3. Run `py main.py`
4. Isi data sesuai dengan prompt yang diberikan.
5. Tunggu model selesai dan hasil keluar :D

## Identitas
- <a href = "https://github.com/LordGedelicious">Gede Prasidha Bhawarnawa (IF 2020 - 13520004)</a>

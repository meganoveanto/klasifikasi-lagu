import joblib
import streamlit as st
import pandas as pd
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

st.title("Judul: Uji Akurasi Klasifikasi Pada Lirik Lagu Berbahasa Indonesia")
st.text("Parameter default SVM : gamma=scale dan C=1.0")
st.text("Hyperparameter PSO : gamma=2.05302752 dan C=7.35185408")

name = st.text_input("Masukkan lirik lagu:")
chkbx = st.checkbox('Gunakan Tuning Hyperparameter PSO')
if chkbx:
    teks = name.title()
    model = joblib.load(open('svm-pso_joblibs.pkl', 'rb'))
    tfidf = joblib.load(open('tfidf-pso_joblibs.pkl', 'rb'))
else:
    teks = name.title()
    model = joblib.load(open('svm_joblib.pkl', 'rb'))
    tfidf = joblib.load(open('tfidf_joblib.pkl', 'rb'))

def preprocessing(lirik):
    lirik = data_cleaning(lirik)
    lirik = case_folding(lirik)
    lirik = normalisasi_kata(lirik)
    lirik = tokenizing(lirik)
    lirik = stopword_kata(lirik)
    lirik = stemming_kata(lirik)
    lirik = rejoin(lirik)
    return lirik

def data_cleaning(lirik):
    data_clean=lirik
    data_clean = re.sub(r'\d+', " ", data_clean)  #remove angka
    data_clean = re.sub(r'http\S+', " ", data_clean)  #hapus url
    data_clean = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\d\*]", " ", data_clean)# Remove simbol, angka dan karakter aneh
    return data_clean

def case_folding(lirik):
    case_fold=lirik.lower()
    return case_fold

def normalisasi_kata(lirik):
    # kamus normalisasi indonesia
    DATA_KBBI = [kamus.strip('\n').strip('\r') for kamus in open('KBBI.txt')]
    dic = {}
    for i in DATA_KBBI:
        (key, val) = i.split('\t')
        dic[str(key)] = val

    kata_awal = lirik.split()
    kata_normalisasi = []
    for i in kata_awal:
        kata_normalisasi.append(dic.get(i, i))
    kata_normalisasi = ' '.join(kata_normalisasi)
    return kata_normalisasi

def tokenizing(lirik):
    tokenisasi=word_tokenize(lirik)
    return tokenisasi

def stopword_kata(lirik):
    # stopword indonesia
    stopword_list = set(stopwords.words('indonesian'))

    list_data=lirik
    stopW = [word for word in list_data if not word in stopword_list]
    return stopW

def stemming_kata(lirik):
    # stemmer indonesia
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factori = StemmerFactory()
    stemmer = factori.create_stemmer()

    list_data=lirik
    stemming = [stemmer.stem(word) for word in list_data]
    return stemming
def rejoin(lirik):
    rejoin_teks=" ".join(lirik)
    rejoin_teks=re.sub(r'\d+', '',  rejoin_teks) #remove number **
    return rejoin_teks

lagu = preprocessing(teks)
data = tfidf.transform([lagu])
hasil = model.predict(data)
hasil1 = " ".join(hasil)
hasil_proba = model.predict_proba(data)
if st.button('Mulai'):
     st.write('**Teks yang dimasukkan :**')
     st.write(lagu)
     st.write('**Hasil kelas emosinya adalah :**',hasil1)
     st.write('**Nilai probabilitas setiap kelas :**')
     df0 = pd.DataFrame(hasil_proba, columns=model.classes_)
     st.dataframe(df0)







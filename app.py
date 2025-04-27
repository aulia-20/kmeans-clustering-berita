
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Fungsi untuk melatih model KMeans
def train_model(news_titles):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(news_titles)
    
    # Elbow method untuk menentukan jumlah cluster optimal
    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)
        
    # Visualisasi Elbow Method
    plt.plot(range(1, 11), sse, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.grid(True)
    st.pyplot()
    
    # Memilih K optimal
    optimal_k = 3  # Misalnya kita pilih 3 (bisa diubah sesuai analisis Elbow)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    kmeans.fit(X)
    return kmeans, vectorizer

# Fungsi untuk memprediksi kategori berdasarkan judul berita
def predict_category(news_title, model, vectorizer):
    news_vec = vectorizer.transform([news_title])
    prediction = model.predict(news_vec)
    return prediction[0]

# Menu Streamlit
st.title('Berita Clustering dengan K-Means')

menu = st.sidebar.selectbox('Pilih Menu', ['Training', 'Predict'])

if menu == 'Training':
    st.subheader('Masukkan judul berita untuk pelatihan model')

    # Contoh data berita (judul berita)
    news_titles = [
        'Pemerintah Indonesia Meluncurkan Program Vaksinasi',
        'Topik Keamanan Siber di Indonesia Meningkat',
        'Peningkatan Ekonomi Pasca Pandemi di Indonesia',
        'Teknologi AI dalam Industri Otomotif',
        'Kebijakan Pajak Baru untuk Bisnis di Indonesia',
        'Meningkatnya Pemakaian Teknologi di Pendidikan',
        'Potensi Energi Terbarukan di Indonesia',
        'Isu Lingkungan Hidup dan Perubahan Iklim',
        'Pendidikan dan Pengembangan Sumber Daya Manusia di Indonesia',
        'Pembangunan Infrastruktur di Seluruh Indonesia'
    ]

    if st.button('Train Model'):
        model, vectorizer = train_model(news_titles)
        st.success('Model berhasil dilatih!')
        st.write('Model siap digunakan untuk prediksi!')

elif menu == 'Predict':
    st.subheader('Masukkan judul berita untuk prediksi kategori')
    news_title = st.text_input('Judul Berita:')
    
    if st.button('Prediksi'):
        if news_title:
            prediction = predict_category(news_title, model, vectorizer)
            st.write(f'Judul berita ini masuk ke cluster: {prediction}')
        else:
            st.warning('Masukkan judul berita terlebih dahulu!')

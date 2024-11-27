import streamlit as st
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Fungsi untuk memuat model dan TfidfVectorizer
def load_model(model_name):
    model_filename = f'{model_name}.pkl'
    if os.path.exists(model_filename):
        with open(model_filename, 'rb') as file:
            model = pickle.load(file)
        return model
    else:
        st.error(f"Model {model_name} tidak ditemukan.")
        return None

# Fungsi untuk memuat TfidfVectorizer
def load_tfidf_vectorizer():
    tfidf_filename = 'tfidfvector.pkl'
    if os.path.exists(tfidf_filename):
        with open(tfidf_filename, 'rb') as file:
            tfidf_vectorizer = pickle.load(file)
        return tfidf_vectorizer
    else:
        st.error("TfidfVector tidak ditemukan.")
        return None

# Antarmuka pengguna Streamlit
st.title("Prediksi Bullying dengan Model Machine Learning")

# Pilih model yang akan digunakan
model_choice = st.selectbox("Pilih Model", ("LinearSVC", "LogisticRegression", "MultinomialNB", 
                                           "DecisionTreeClassifier", "AdaBoostClassifier", 
                                           "BaggingClassifier", "SGDClassifier"))

# Memuat model yang dipilih
model = load_model(model_choice)
if model is None:
    st.stop()

# Memuat TfidfVectorizer
tfidf_vectorizer = load_tfidf_vectorizer()
if tfidf_vectorizer is None:
    st.stop()

# Input untuk memasukkan tweet
tweet = st.text_area("Masukkan tweet yang akan diuji:", "")

# Prediksi ketika tombol ditekan
if st.button("Prediksi"):
    if tweet.strip() == "":
        st.warning("Harap masukkan tweet terlebih dahulu!")
    else:
        # Mengubah tweet menjadi representasi TF-IDF
        tweet_vectorized = tfidf_vectorizer.transform([tweet])

        # Melakukan prediksi menggunakan model yang dipilih
        prediction = model.predict(tweet_vectorized)

        if prediction == 1:
            st.write("Tweet ini **mengandung bullying**.")
        else:
            st.write("Tweet ini **tidak mengandung bullying**.")
        
        # Menampilkan metrik performa (optional)
        st.write("Model Performance (on test set):")
        st.write(f"Accuracy: {model.score(x_test, y_test):.2f}")
        st.write(f"F1 Score: {f1_score(y_test, model.predict(x_test), average='weighted'):.2f}")
        st.write(f"Precision: {precision_score(y_test, model.predict(x_test), average='weighted'):.2f}")
        st.write(f"Recall: {recall_score(y_test, model.predict(x_test), average='weighted'):.2f}")


import streamlit as st
import joblib
import re
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ------------------- DOWNLOAD NLTK DATA -------------------
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ------------------- LOAD MODEL & ARTIFAK -------------------
@st.cache_resource
def load_artifacts():
    tfidf      = joblib.load("model/tfidf_vectorizer.pkl")
    model      = joblib.load("model/logreg_model.pkl")
    slangwords = joblib.load("model/slangwords.pkl")
    return tfidf, model, slangwords

tfidf, model, slangwords = load_artifacts()

# ------------------- PRE-PROCESSING FUNGSI -------------------
def cleaningText(text):
    text = re.sub(r'@[A-Za-z0-9]+','',text)     # remove mentions
    text = re.sub(r'#[A-Za-z0-9]+','',text)     # remove hashtag
    text = re.sub(r'RT[\s]','',text)            # remove RT
    text = re.sub(r"http\S+",'',text)           # remove link
    text = re.sub(r'[0-9]+','',text)            # remove numbers
    text = re.sub(r'[^\w\s]','',text)           # remove punctuation
    text = text.replace('\n',' ')               # replace newline
    text = text.translate(str.maketrans('','',string.punctuation))
    text = text.strip()
    return text

def casefoldingText(text):
    return text.lower()

def fix_slangwords(text):
    words = text.split()
    fixed = [slangwords.get(w.lower(), w) for w in words]
    return ' '.join(fixed)

def tokenizingText(text):
    return word_tokenize(text)

def filteringText(tokens):
    stop_words = set(stopwords.words('indonesian') + stopwords.words('english'))
    extra = {'iya','yaa','gak','nya','na','sih','ku',"di","ga","ya","gaa","loh","kah"}
    stop_words.update(extra)
    return [w for w in tokens if w not in stop_words]

def toSentence(tokens):
    return ' '.join(tokens)

def preprocess(text):
    txt = cleaningText(text)
    txt = casefoldingText(txt)
    txt = fix_slangwords(txt)
    tokens = tokenizingText(txt)
    tokens = filteringText(tokens)
    return toSentence(tokens)

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="Analisis Sentimen Aplikasi by.u", layout="centered")

st.title("Analisis Sentimen Ulasan by.u")
st.caption("Masukkan satu ulasan, lalu tekan **Predict**")

user_input = st.text_area("Ulasan", height=150,
                          placeholder="Contoh: aplikasinya bagus banget, cepet, gampang topup")

if st.button("Predict", type="primary"):
    if not user_input.strip():
        st.warning("Tulis ulasan dulu ya!")
    else:
        with st.spinner("Sedang memproses..."):
            clean_text = preprocess(user_input)
            vec        = tfidf.transform([clean_text])
            pred       = model.predict(vec)[0]
            score      = model.predict_proba(vec)[0].max()

        st.success("**SELESAI!**")
        col1, col2 = st.columns([1,2])
        with col1:
            st.metric("Prediksi", pred.upper())
        with col2:
            st.progress(score)
            st.caption(f"Confidence: {score:.1%}")

        with st.expander("Lihat proses preprocessing"):
            st.write("Teks bersih →", clean_text)
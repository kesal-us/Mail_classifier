import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

# Load the models
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Set Page Configuration
st.set_page_config(page_title="Disease Prediction", page_icon="📧", layout="wide")

# Hiding Streamlit UI Elements
hide_st_style = """
    <style>
    /* Remove Streamlit menu, footer, and header */
    #MainMenu {display: none !important;}
    footer {display: none !important;}
    header {display: none !important;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# Function to preprocess text
def transform(text):
    text = text.lower()  # Lowercase
    text = nltk.word_tokenize(text)  # Tokenization

    # Remove special characters
    text = [i for i in text if i.isalnum()]

    # Remove stopwords and punctuations
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Stemming
    ps = PorterStemmer()
    text = [ps.stem(i) for i in text]

    return ' '.join(text)

# Page styling
st.markdown(
    """
    <style>
        body {
            background-color: #f0f2f6;
        }
        .main {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
        .title {
            color: #ff4b4b;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
        }
        .result {
            font-size: 40px;
            text-align: center;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# App UI
st.markdown('<h1 class="title">📧 Email Spam Classifier</h1>', unsafe_allow_html=True)
st.write("### Enter an email to check if it's spam or not:")

# Input field
input_mail = st.text_area("", height=150)

# Predict button
if st.button('🔎 Predict', key="predict"):
    if input_mail.strip() == "":
        st.warning("⚠️ Please enter an email!")
    else:
        # Preprocess
        new = transform(input_mail)

        # Vectorize
        vec = tfidf.transform([new])

        # Predict
        res = model.predict(vec)

        # Display result
        if res == 1:
            st.markdown('<p class="result" style="color:red;">🚨 This email is SPAM!</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="result" style="color:green;">✅ This email is NOT spam.</p>', unsafe_allow_html=True)

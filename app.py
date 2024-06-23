import streamlit as st
import pandas as pd
import pickle
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer


nltk.download('punkt')
nltk.download('stopwords')


html_temp_subtitle = """
    <div style="background-color:#ff6347;padding:10px;margin-bottom:20px">
    <h2 style="color:white;text-align:center;">SPAM Detection</h2>
    </div>
    """
st.markdown(html_temp_subtitle, unsafe_allow_html=True)

st.subheader('Enter Text')


ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('spam_classifier.pkl','rb'))

st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    transformed_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        st.header("Spam")
    elif result==0:
        st.header("Not Spam")
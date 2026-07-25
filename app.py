import streamlit as st
import joblib

model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

st.title("movie sentiment analysis")
review= st.text_area("enter your review")
if st.button("predict"):
  review_tfidf= tfidf.transform([review])
  prediction=model.predict(review_tfidf)
  st.write("prediction:",prediction[0])
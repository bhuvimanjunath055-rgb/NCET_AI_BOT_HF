import streamlit as st
from transformers import pipeline

st.title("AI Text Summarizer")

text = st.text_area("Enter text")

if st.button("Summarize"):
    if text == "":
        st.write("Enter text")
    else:
        try:
            summarizer = pipeline("summarization", model="t5-small", device=-1)
            result = summarizer(text[:200])
            st.write(result[0]['summary_text'])
        except Exception as e:
            st.write("Error:", e)

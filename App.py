import streamlit as st
from transformers import pipeline

st.title("🤖 AI Text Summarizer")
st.write("Enter a long text below, and get a concise summary!")

# Input
long_text = st.text_area("Enter text to summarize:", height=200)

# Parameters
max_length = st.slider("Max Summary Length", 50, 300, 120)
min_length = st.slider("Min Summary Length", 20, 100, 30)

# Button
if st.button("Summarize"):
    if not long_text.strip():
        st.warning("Please enter some text!")
    else:
        try:
            with st.spinner("Loading model... please wait ⏳"):
                summarizer = pipeline("summarization", model="t5-small")

            with st.spinner("Generating summary..."):
                summary = summarizer(
                    long_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )

            st.success("Summary:")
            st.write(summary[0]['summary_text'])

        except Exception as e:
            st.error(f"Error: {e}")

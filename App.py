import streamlit as st
from transformers import pipeline

# Load summarizer (lightweight model to avoid crash)
@st.cache_resource
def load_summarize():
    try:
        return pipeline("summarization", model="t5-small")
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

summarizer = load_summarize()

# UI
st.title("🤖 AI Text Summarizer")
st.write("Enter a long text below, and get a concise summary!")

# Input
long_text = st.text_area("Enter text to summarize:", height=200)

# Parameters
max_length = st.slider("Max Summary Length", 50, 300, 120)
min_length = st.slider("Min Summary Length", 20, 100, 30)

# Button
if st.button("Summarize"):
    if summarizer is None:
        st.error("Model not loaded properly.")
    elif not long_text.strip():
        st.warning("Please enter some text!")
    else:
        try:
            summary = summarizer(
                long_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            st.success("Summary:")
            st.write(summary[0]['summary_text'])
        except Exception as e:
            st.error(f"Error during summarization: {e}")

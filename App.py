import streamlit as st
from transformers import pipeline

# Load the summarization model safely
@st.cache_resource
def load_summarize():
    try:
        return pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6"
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

summarizer = load_summarize()

# UI
st.title("🤖 AI Text Summarizer")
st.write("Enter a long text below, and get a concise summary!")

# Input
long_text = st.text_area("Enter text to summarize:", height=200)

# Sliders
max_length = st.slider("Max Summary Length", 50, 300, 130)
min_length = st.slider("Min Summary Length", 20, 100, 30)

# Button
if st.button("Summarize"):
    if summarizer is None:
        st.error("Model not loaded. Please check deployment settings.")
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
            st.error(f"Summarization failed: {e}")

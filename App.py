import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Text Summarizer")

st.title("🤖 AI Text Summarizer")

@st.cache_resource
def load_model():
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        device=-1  # CPU only (important for Streamlit Cloud)
    )

summarizer = load_model()

text = st.text_area("Enter text to summarize")

if st.button("Summarize"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        try:
            result = summarizer(
                text,
                max_length=120,
                min_length=30,
                do_sample=False
            )
            st.success("Summary:")
            st.write(result[0]["summary_text"])
        except Exception as e:
            st.error(f"Error: {e}")

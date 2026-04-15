import streamlit as st
from transformers import pipeline

st.title("🤖 AI Text Summarizer")
st.write("Enter text below to summarize")

# Load model safely (CPU only)
@st.cache_resource
def load_model():
    try:
        return pipeline(
            "summarization",
            model="t5-small",
            device=-1   # 🔴 force CPU (important)
        )
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None

summarizer = load_model()

# Input
long_text = st.text_area("Enter text:", height=200)

# Controls
max_length = st.slider("Max Summary Length", 30, 120, 60)
min_length = st.slider("Min Summary Length", 10, 60, 20)

# Button
if st.button("Summarize"):

    if summarizer is None:
        st.error("Model not loaded.")
    
    elif not long_text.strip():
        st.warning("Please enter some text!")

    else:
        try:
            # 🔴 Limit input size (VERY IMPORTANT)
            input_text = long_text[:300]

            with st.spinner("Summarizing..."):
                summary = summarizer(
                    input_text,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=False
                )

            st.success("Summary:")
            st.write(summary[0]['summary_text'])

        except Exception as e:
            st.error(f"Error during summarization: {e}")

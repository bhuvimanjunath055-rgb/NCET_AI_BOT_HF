import streamlit as st
from transformers import pipeline

st.title("🤖 AI Text Summarizer")
st.write("Enter text below to summarize")

# Load model ONCE safely
@st.cache_resource
def load_model():
    try:
        return pipeline("summarization", model="t5-small")
    except Exception as e:
        return None

summarizer = load_model()

# Input
long_text = st.text_area("Enter text:", height=200)

# Sliders
max_length = st.slider("Max Summary Length", 30, 150, 80)
min_length = st.slider("Min Summary Length", 10, 80, 30)

# Button
if st.button("Summarize"):

    if summarizer is None:
        st.error("Model failed to load.")
    
    elif not long_text.strip():
        st.warning("Please enter some text!")

    else:
        try:
            # 🔴 IMPORTANT: limit input size (prevents crash)
            input_text = long_text[:500]

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
            st.error(f"Error: {e}")

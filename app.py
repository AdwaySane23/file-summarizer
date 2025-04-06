import os
# Force the use of the pure-Python implementation of tokenizers.
os.environ["TOKENIZERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import tempfile
import PyPDF2
import docx
from transformers import pipeline

def extract_text(file):
    """
    Extract text from various file types. For file-like objects, we use file.name 
    to determine the extension.
    """
    filename = file.name
    ext = os.path.splitext(filename)[1].lower()
    
    if ext in ['.txt', '.md']:
        # For text files, decode bytes to string
        return file.read().decode('utf-8')
    
    elif ext == '.pdf':
        # PyPDF2 works with file-like objects
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    
    elif ext == '.docx':
        # python-docx does not support file-like objects directly, so we save temporarily.
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        doc = docx.Document(tmp_path)
        text = "\n".join([para.text for para in doc.paragraphs])
        os.unlink(tmp_path)
        return text
    else:
        return None

def chunk_text(text, chunk_size=500):
    """Split text into chunks to handle model input limits."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

@st.cache_resource(show_spinner=False)
def get_summarizer():
    """Load and cache the summarization pipeline."""
    return pipeline("summarization")

def summarize_text(text):
    summarizer = get_summarizer()
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        # Adjust max_length/min_length as needed
        summary = summarizer(chunk, max_length=150, min_length=40, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return "\n\n".join(summaries)

# Streamlit App Layout
st.title("File Summarizer")
st.write("Upload a file (.txt, .pdf, or .docx) to get a summary of its content.")

uploaded_file = st.file_uploader("Choose a file", type=["txt", "md", "pdf", "docx"])

if uploaded_file is not None:
    text = extract_text(uploaded_file)
    if text is None:
        st.error("Unsupported file type.")
    else:
        st.write("Extracted text length:", len(text))
        st.info("Summarizing... please wait.")
        summary = summarize_text(text)
        st.subheader("Summary")
        st.write(summary)

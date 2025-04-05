import streamlit as st
import PyPDF2
from transformers import pipeline

# Cache the pipeline so that it loads only once.
@st.cache_resource(show_spinner=False)
def get_summarizer():
    # Explicitly specify the model and revision to avoid internal API issues.
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",
        revision="a4f8f3e",  # optional: specify revision if needed
    )
    return summarizer

def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
    except Exception as e:
        st.error(f"Error opening PDF file: {e}")
        return None

    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

st.title("File Summarizer")

uploaded_file = st.file_uploader("Choose a file", type=["txt", "md", "pdf", "docx"])

if uploaded_file is not None:
    # Process the file based on its type.
    if uploaded_file.type == "application/pdf":
        text = extract_text_from_pdf(uploaded_file)
    elif uploaded_file.type in ["text/plain", "text/markdown"]:
        text = uploaded_file.getvalue().decode("utf-8")
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # For Word documents, use python-docx to extract text.
        from docx import Document
        doc = Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        text = None

    if text:
        st.write("Extracted text length:", len(text))
        st.info("Summarizing... please wait.")
        summarizer = get_summarizer()
        # Optionally, you might split long text into chunks if needed.
        summary_result = summarizer(text, max_length=130, min_length=30, do_sample=False)
        st.subheader("Summary")
        st.write(summary_result[0]['summary_text'])
    else:
        st.error("Unsupported file type or unable to extract text.")

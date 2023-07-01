# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import docx
import pdfplumber
import io
from summarizer import Summarizer

# Initialize sentence transformer model for BERT embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from DOC file
def extract_text_from_doc(doc_file):
    doc = docx.Document(doc_file)
    text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    return text

# Function to extract text from PDF file
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ' '.join([page.extract_text() for page in pdf.pages])
    return text

# Function to summarize text
def summarize_text(text):
    model = Summarizer()
    return model(text, min_length=60, max_length=150)

# Function to calculate similarity between resumes and job description
def calculate_similarity(resumes, job_desc):
    # Calculate BERT embeddings
    resume_embeddings = model.encode(resumes, convert_to_tensor=True)
    job_desc_embedding = model.encode([job_desc], convert_to_tensor=True)

    # Calculate cosine similarity between resumes and job description
    similarities = cosine_similarity(resume_embeddings, job_desc_embedding)

    return similarities

# Main application
def main():
    st.title('Resume Analyzer and Job Matcher')

    # Allow user to upload resumes
    uploaded_file = st.file_uploader("Upload Resumes", type=['doc', 'docx', 'pdf'])
    job_desc = st.text_area("Enter Job Description")

    if uploaded_file is not None and job_desc:
        if uploaded_file.type == 'application/pdf':
            resume_text = extract_text_from_pdf(io.BytesIO(uploaded_file.getvalue()))
        elif uploaded_file.type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
            resume_text = extract_text_from_doc(io.BytesIO(uploaded_file.getvalue()))

        # Calculate similarity
        similarities = calculate_similarity([resume_text], job_desc)

        # Display similarity
        st.write(f"Resume match score with job description: {similarities[0][0]}")
        
        # Summarize resume and display it
        summary = summarize_text(resume_text)
        st.write(f"Resume Summary: {summary}")

if __name__ == "__main__":
    main()

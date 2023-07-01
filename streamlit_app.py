# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import docx
import PyPDF2

# Initialize sentence transformer model for BERT embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from DOC file
def extract_text_from_doc(doc_path):
    doc = docx.Document(doc_path)
    text = ' '.join([paragraph.text for paragraph in doc.paragraphs])
    return text

# Function to extract text from PDF file
def extract_text_from_pdf(pdf_path):
    pdf_file_obj = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfFileReader(pdf_file_obj)
    text = ' '.join([pdf_reader.getPage(i).extract_text() for i in range(pdf_reader.numPages)])
    pdf_file_obj.close()
    return text

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
            resume_text = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type in ['application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'application/msword']:
            resume_text = extract_text_from_doc(uploaded_file)

        # Calculate similarity
        similarities = calculate_similarity([resume_text], job_desc)

        # Display similarity
        st.write(f"Resume match score with job description: {similarities[0][0]}")

if __name__ == "__main__":
    main()

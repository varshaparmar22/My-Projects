import streamlit as st
import boto3
import uuid

# ---------------------------
# import necessary libraries & variables
# ---------------------------
import os
from dotenv import load_dotenv


# ---------------------------
# Streamlit app for PDF text extraction using AWS Textract  
# AWS Config
# ---------------------------

load_dotenv()
REGION = os.getenv("AWS_REGION", "us-east-1")
BUCKET_NAME = os.getenv("S3_BUCKET") 

s3 = boto3.client('s3', region_name=REGION)
textract = boto3.client('textract', region_name=REGION)

def upload_to_s3(file):
    file_id = str(uuid.uuid4()) + ".pdf"
    s3.upload_fileobj(file, BUCKET_NAME, file_id)
    return file_id

def extract_text_from_textract_s3(file_name):
    response = textract.detect_document_text(
        Document={'S3Object': {'Bucket': BUCKET_NAME, 'Name': file_name}}
    )
    lines = [item['Text'] for item in response['Blocks'] if item['BlockType'] == 'LINE']
    return '\n'.join(lines)

# Streamlit UI
st.title("📄 PDF Text Extractor (AWS Textract Free Tier)")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Uploading and processing..."):
        file_name = upload_to_s3(uploaded_file)
        extracted_text = extract_text_from_textract_s3(file_name)
    
    st.success("✅ Text extracted!")
    st.text_area("Extracted Text", extracted_text, height=300)

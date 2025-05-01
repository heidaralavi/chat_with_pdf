# Install required packages
# pip install python-dotenv requests pymupdf python-bidi arabic-reshaper

import os
import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bidi.algorithm import get_display
import arabic_reshaper
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
import pymupdf
from arabic_reshaper import reshape
import pdfplumber


# Configuration
class Config:
    HF_API_TOKEN = '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'
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    QA_MODEL = "HooshvareLab/bert-fa-base-uncased"
    PDF_PATH = "example.pdf"


def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extract text and handle encoding
            page_text = page.extract_text()
            if page_text:
                text += reshape(page_text) 
    return text

pdf_text = extract_text_from_pdf(Config.PDF_PATH)
#print(pdf_text)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=5,
    separators=['\n']         #"\n\n", "\n", "۔", "؟", "!", " "]
    )

texts = text_splitter.split_text(pdf_text)

print(len(texts))
#rtl_text = get_display(reshaped_text)


embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=Config.HF_API_TOKEN,
            model_name=Config.EMBEDDING_MODEL
        )


vectorstore = FAISS.from_texts(texts, embeddings)

question = reshape('آکواریوم خاطرات کجا است')

#docs = vectorstore.similarity_search(question, k=3)
docs = vectorstore.similarity_search_with_score(question, k=3)

print(len(docs))


#print(docs[0][0], docs[0][1])
print(docs[1][0], docs[1][1])
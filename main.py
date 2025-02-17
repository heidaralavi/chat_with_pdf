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

# Load environment variables
load_dotenv()

# Configuration
class Config:
    HF_API_TOKEN = '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'
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    QA_MODEL = "HooshvareLab/bert-fa-base-uncased"
    PDF_PATH = "example.pdf"

# Persian text formatting
def format_persian(text):
    reshaped = arabic_reshaper.reshape(text)
    #print(get_display(reshaped))
    return get_display(reshaped)

# PDF processing pipeline
class PersianPDFChatter:
    def __init__(self):
        self.embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=Config.HF_API_TOKEN,
            model_name=Config.EMBEDDING_MODEL
        )
        
        # Load and process PDF
        loader = PyMuPDFLoader(Config.PDF_PATH)
        documents = loader.load()
        print(documents)
        
        # Persian-optimized text splitting
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "۔", "؟", "!", " "]
        )
        self.texts = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = FAISS.from_documents(self.texts, self.embeddings)
    
    def ask_question(self, question):
        # Find relevant context
        docs = self.vectorstore.similarity_search(format_persian(question), k=3)
        context = "\n".join([doc.page_content for doc in docs])
        print(context)
        # Query HF API
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{Config.QA_MODEL}",
            headers={"Authorization": f"Bearer {Config.HF_API_TOKEN}"},
            json={
                "inputs": {
                    "question": format_persian(question),
                    "context": format_persian(context)
                }
            }
        )
        #print(response.json())
        return response.json().get('answer', format_persian("پاسخی یافت نشد"))

# Chat interface
def main():
    chatter = PersianPDFChatter()
    
    print(format_persian("خوش آمدید! برای خروج 'خروج' تایپ کنید"))
    
    while True:
        question = input(format_persian("\nسوال شما: "))
        if question.strip().lower() == format_persian('خروج'):
            break
            
        answer = chatter.ask_question(question)
        print(format_persian(f"\nپاسخ: {answer}"))

if __name__ == "__main__":
    main()
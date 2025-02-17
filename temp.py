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

# Configuration
class Config:
    HF_API_TOKEN = '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'
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    QA_MODEL = "HooshvareLab/bert-fa-base-uncased"
    PDF_PATH = "example.pdf"


# Load and process PDF
loader = PyMuPDFLoader(Config.PDF_PATH)
documents = loader.load()
#reshaped = arabic_reshaper.reshape(pages[0].page_content)

# Persian-optimized text splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "۔", "؟", "!", " "]
    )
texts = text_splitter.split_documents(documents)

#print(get_display(texts[0].page_content))

embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=Config.HF_API_TOKEN,
            model_name=Config.EMBEDDING_MODEL
        )


vectorstore = FAISS.from_documents(texts, embeddings)

#print(vectorstore)
question = 'نام کهکشان چیست'

docs = vectorstore.similarity_search('نام کهکشان چیست', k=3)

#print([get_display(doc.page_content) for doc in docs])
context = [get_display(doc.page_content).replace('\n','') for doc in docs]
print(type(context))
tt = arabic_reshaper.reshape(context[0])
response = requests.post(
            f"https://api-inference.huggingface.co/models/{Config.QA_MODEL}",
            headers={"Authorization": f"Bearer {Config.HF_API_TOKEN}"},
            json={
                "inputs": {
                    "question": 'دوستان چه ',
                    "context": 'سلام دوستان حال شما چطور است'
                }
            }
        )

print(response.json())

#reshaped = arabic_reshaper.reshape(texts[0])
#print(reshaped)
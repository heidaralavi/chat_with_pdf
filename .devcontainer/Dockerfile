FROM python:3.10-slim
USER root
SHELL ["/bin/bash", "-c"]
RUN apt-get update && apt-get install curl ffmpeg libsm6 tesseract-ocr libxext6 poppler-utils fontconfig -y && pip install --upgrade pip
RUN curl -fsSL https://ollama.com/install.sh | sh
COPY /fonts /usr/share/fonts
COPY /nltk_data /root/nltk_data
COPY pip.config /usr/local/bin/
RUN fc-cache -f -v
EXPOSE 8888
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
RUN ollama serve & sleep 5 && ollama pull nomic-embed-text
RUN ollama serve & sleep 5 && ollama pull llama3.2
COPY . .



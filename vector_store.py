from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyCy-F4waBhpZEwUeT5FH-ulfOT_0ySKOXw"


def create_vector_store(documents):
    # embeddings = VertexAIEmbeddings()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Tạo danh sách văn bản từ các tài liệu
    texts = [doc.page_content for doc in documents]

    # Tạo Chroma vector store từ các embedding đã có
    vector_store = Chroma.from_texts(texts, embeddings)
    return vector_store


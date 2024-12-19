from langchain_community.document_loaders import TextLoader

def load_documents(directory_path):
    loader = TextLoader(directory_path, encoding='utf-8')  # Chỉ định mã hóa UTF-8
    documents = loader.load()
    return documents


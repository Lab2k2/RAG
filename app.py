from document_loader import load_documents
from vector_store import create_vector_store
from llm_model import llm
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
def get_answer_from_txt(query, vector_store, llm):

    prompt_template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know. Answer in Vietnamese

    Context:
    {context}

    Question: {question}

    Answer in a concise manner:
    """
    prompt = PromptTemplate(
        input_variables=[
            "context",
            "question",
        ],  
        template=prompt_template,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=retriever, chain_type_kwargs={"prompt": prompt}
    )

    response = qa_chain.run(query)

    print(f"\r\n\r\n *********query = {query}")
    print(f"LLM Response with Retrieved Context: {response}")

def main():
    # Bước 1: Tải tài liệu
    documents = load_documents("data/input.txt")
    print("==========Load dữ liệu: done============")
    # Bước 2: Tạo vector store
    vector_store = create_vector_store(documents)
    print("==========Tạo Vector: done============")
    # Thực hiện truy vấn
    while True:
        query = input("Nhập câu hỏi: ")
        
        # Kiểm tra nếu câu hỏi là "Tôi hết câu hỏi rồi" thì dừng
        if query.lower() == "tôi hết câu hỏi rồi":
            print("Kết thúc chương trình.")
            break
        
        print(f"Câu hỏi của bạn là: {query}")
        
        # Gọi hàm để lấy câu trả lời từ văn bản
        get_answer_from_txt(query, vector_store, llm)

if __name__ == "__main__":
    main()

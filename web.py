import streamlit as st
from document_loader import load_documents
from vector_store import create_vector_store
from llm_model import llm
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Hàm lấy câu trả lời từ văn bản
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
        input_variables=["context", "question"],
        template=prompt_template,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=retriever, chain_type_kwargs={"prompt": prompt}
    )

    response = qa_chain.run(query)
    return response

# Streamlit
def main():
    st.title("Hệ Thống Hỏi Đáp Dựa Trên Văn Bản")
    
    # Tạo hai cột để chia màn hình
    col1, col2 = st.columns([2, 1])  # Cột trái chiếm 2 phần và cột phải chiếm 1 phần

    # Xử lý file văn bản
    with col1:
        uploaded_file = st.file_uploader("Tải tài liệu (txt hoặc docx)", type=["txt", "docx"])
        
        if uploaded_file is not None:
            # Lưu file tạm thời
            with open("temp_input_file", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Bước 1: Tải tài liệu từ file
            documents = load_documents("temp_input_file")
            st.success("Tải dữ liệu thành công!")

            # Bước 2: Hiển thị nội dung tài liệu (có thanh cuộn nếu nội dung dài)
            st.subheader("Nội dung tài liệu:")
            with open("temp_input_file", "r", encoding="utf-8") as f:
                content = f.read()
            st.text_area("Nội dung tài liệu", content, height=400)  # Thanh cuộn cho nội dung

        else:
            st.info("Vui lòng tải lên một tài liệu.")

    # Nhập câu hỏi và trả lời
    with col2:
        if uploaded_file is not None:
            # Bước 1: Tạo vector store từ văn bản
            vector_store = create_vector_store(documents)
            
            # Bước 2: Nhập câu hỏi và tìm câu trả lời
            query = st.text_input("Nhập câu hỏi:")
            
            if query:
                if query.lower() == "tôi hết câu hỏi rồi":
                    st.write("Kết thúc chương trình.")
                else:
                    st.write(f"Câu hỏi của bạn: {query}")
                    
                    # Gọi hàm lấy câu trả lời từ văn bản
                    response = get_answer_from_txt(query, vector_store, llm)
                    
                    st.write(f"Câu trả lời: {response}")
    
if __name__ == "__main__":
    main()

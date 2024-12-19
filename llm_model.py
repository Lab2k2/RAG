import os
#from langchain_google_vertexai import ChatVertexAI

from langchain_google_genai import  ChatGoogleGenerativeAI
def load_llm():
    os.environ["GOOGLE_API_KEY"] = ("AIzaSyCy-F4waBhpZEwUeT5FH-ulfOT_0ySKOXw")  # Thay bằng API Key thực tế của bạn
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    return llm

llm = load_llm()

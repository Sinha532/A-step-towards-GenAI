import streamlit as st
import os
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import tempfile

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.7,
    convert_system_message_to_human=True
)

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding(pdf_path):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    final_documents = text_splitter.split_documents(docs[:20])
    vectors = FAISS.from_documents(final_documents, embeddings)
    return vectors

def retrieval(question, pdf_path):
    vectors = vector_embedding(pdf_path)
    doc_chain = create_stuff_documents_chain(llm,prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, doc_chain)
    result = retrieval_chain.invoke({'input': question})
    return result['answer']

st.markdown(
    """
    <h1 style='text-align: center;'>Chat with your PDF</h1>
    """,
    unsafe_allow_html=True
)
st.markdown("""
            <h5 style='text-align: center;'> Upload a document below and ask a question about it. Relax -Let us answer it!</h5>
            """,
            unsafe_allow_html=True)

page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

file_upload = st.file_uploader("Upload the PDF here", type="pdf")

if file_upload is not None:
    question = st.text_input("Ask your question here")

    if question:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_upload.read())
            temp_pdf_path = temp_file.name

        # Pass the temporary file path to the retrieval function
        answer = retrieval(question, temp_pdf_path)
        st.write(answer)

        # Clean up the temporary file after use
        os.remove(temp_pdf_path)

import streamlit as st
import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

st.title("📚 Multi-PDF Chatbot")

# List of PDF files
PDF_FILES = [
    "converted_text.pdf",
    "robert_greene.pdf"
    # Add more PDF paths here
]

@st.cache_resource
def load_multiple_pdfs():

    all_docs = []

    for pdf_path in PDF_FILES:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_docs.extend(documents)

    text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    docs = text_splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(all_docs, embeddings)

    return vectorstore

db = load_multiple_pdfs()
query = st.text_input("Ask something about your document : ")

if query : 
    docs = db.similarity_search(query, k=3)
    llm = OpenAI(temperature = 0.7)
    chain = load_qa_chain(llm, chain_type="stuff")

    response = chain.run(input_documents = docs, question = query)
    st.write(response)
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 09:20:11 2024

@author: Rise Networks
"""
import sys
__import__('pysqlite3')
import pysqlite3
sys.modules['sqlite3'] = sys.modules["pysqlite3"]
import chromadb
import tempfile
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv
from uuid import uuid4
import  shutil

# Load environment variables
load_dotenv()
# Groq_API_KEY
os.getenv("GROQ_API_KEY")
os.getenv("GOOGLE_API_KEY")

# File Uploading
uploaded_file = st.sidebar.file_uploader(
    "Upload document", ["pdf", "docx", "txt"]
)

if uploaded_file != None:
    st.sidebar.success("Document Uploaded", icon="✔️")
else:
    st.sidebar.warning("Upload a document", icon="⚠️")

# Creating a static header
st.header("Chat with Document 🤖")
    
# Creating the chat box
container = st.container(border=True, height=300,)

question = st.chat_input("Ask a question")

# create a function to confirm the document name
def confirm_document_name(document_name):
    if document_name[-3:] == "pdf":
        with tempfile.NamedTemporaryFile(suffix="pdf", delete=False) as file:
            file.write(uploaded_file.getbuffer())
            tempfile_path = file.name
    elif document_name[-3:] == "txt":
        with tempfile.NamedTemporaryFile(suffix="txt", delete=False) as file:
            file.write(uploaded_file.getbuffer())
            tempfile_path = file.name
    else: 
        with tempfile.NamedTemporaryFile(suffix="docx", delete=False) as file:
            file.write(uploaded_file.getbuffer())
            tempfile_path = file.name
    #return document name
    return tempfile_path

# create a function to load the document
def load_document(document_name):
    file_name = confirm_document_name(document_name=document_name)
    if file_name[-3:] == "pdf":
        loader = PyPDFLoader(file_name)
    elif file_name[-3:] == "txt":
        loader = TextLoader(file_name)
    else:
        loader = Docx2txtLoader(file_name)
    # Load the document
    doc = loader.load()
    return doc

# Split the document
def split_document(document):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=1500,
        chunk_overlap=150,
    )
    
    # splitting the document
    chunks = text_splitter.split_documents(document)
    return chunks

# Creating a database
def create_db(embeddings, chunks):
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="db",
        #collection_name="PDF_chat"
    )
    return db.as_retriever()
    
      
# define app
def app():
    qa = None
    if uploaded_file:
        document_name = uploaded_file.name
        
        #The file path of the document
        doc_path = confirm_document_name(document_name=document_name)
        
        # The document itself being loaded
        document = load_document(document_name=document_name)
        
        # the document being splitted
        chunks = split_document(document=document)

        # Initialize our embedding
        os.getenv("GOOGLE_API_KEY")
        embeddings =  GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                                   google_api_key="AIzaSyBSwnx3RH_HCYV0lVUJp1pyx8baRgFKGw4"
                                                  )

        # Creating a database
        vector_db = create_db(embeddings, chunks)
        container.write([embeddings])
        
        os.getenv("GROQ_API_KEY")
        # Creating LLM using Groq-AI
        llm = ChatGroq(
            model="llama3-70b-8192",
            temperature=0,
            max_retries=2,
            max_tokens=1024
        )

        # Creating a Reqtrieval QA
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type = "stuff",
            retriever = vector_db,
            return_source_documents = True,
        )
    return qa
    
if question:
    retrieval_qa = app()
    response = retrieval_qa.invoke(f"{question}")
    container.write(f"{question}")   
    container.write(response["result"])

    
if __name__=="__main__":
    app()

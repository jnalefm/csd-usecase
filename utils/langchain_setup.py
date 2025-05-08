import os
import pathlib
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from utils.prompt import get_prompt
# import keys
from google.generativeai import configure as google_configure
import streamlit as st

# os.environ["GOOGLE_API_KEY"] = keys.GOOGLE_API_KEY
google_configure(api_key=st.secrets.GOOGLE_API_KEY)

VECTORSTORE_DIR = "vectorstores"

def create_vectorstore(pdf_path, product_name):
    vectorstore_path = os.path.join(VECTORSTORE_DIR, product_name.replace(" ", "_"))

    # If vectorstore exists, load it
    if os.path.exists(vectorstore_path):
        return FAISS.load_local(
            vectorstore_path,
            GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
            allow_dangerous_deserialization=True
        )

    # Else, create vectorstore
    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="gradient")
    docs = text_splitter.split_documents(pages)
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save vectorstore
    pathlib.Path(VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(vectorstore_path)

    # Show message that embeddings were updated
    import streamlit as st
    st.success(f"---Embeddings updated for {product_name}---")

    return vectorstore


def get_qa_chain(vectorstore, product_name):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        temperature=0.5,
        max_output_tokens=3048
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    prompt = get_prompt(product_name)

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False
    )
    return chain

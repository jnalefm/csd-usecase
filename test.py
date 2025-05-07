# # for linux
# # !apt-get install poppler-utils tesseract-ocr libmagic-dev

# # %pip install -Uq "unstructured[all-docs]" pillow lxml pillow
# # %pip install -Uq chromadb tiktoken
# # %pip install -Uq langchain langchain-community langchain-openai langchain-groq
# # %pip install -Uq python_dotenv
# # %pip install -Uq langchain-groq
# # %pip install -Uq langchain_openai

# from unstructured.partition.pdf import partition_pdf

# output_path = "./content/"
# file_path = output_path + 'attention.pdf'

# # Reference: https://docs.unstructured.io/open-source/core-functionality/chunking
# chunks = partition_pdf(
#     filename=file_path,
#     infer_table_structure=True,            # extract tables
#     strategy="hi_res",                     # mandatory to infer tables

#     extract_image_block_types=["Image"],   # Add 'Table' to list to extract image of tables
#     # image_output_dir_path=output_path,   # if None, images and tables will saved in base64

#     extract_image_block_to_payload=True,   # if true, will extract base64 for API usage

#     chunking_strategy="by_title",          # or 'basic'
#     max_characters=10000,                  # defaults to 500
#     combine_text_under_n_chars=2000,       # defaults to 0
#     new_after_n_chars=6000,

#     # extract_images_in_pdf=True,          # deprecated
# )

# # separate tables from texts
# tables = []
# texts = []

# for chunk in chunks:
#     if "Table" in str(type(chunk)):
#         tables.append(chunk)

#     if "CompositeElement" in str(type((chunk))):
#         texts.append(chunk)

# # Get the images from the CompositeElement objects
# def get_images_base64(chunks):
#     images_b64 = []
#     for chunk in chunks:
#         if "CompositeElement" in str(type(chunk)):
#             chunk_els = chunk.metadata.orig_elements
#             for el in chunk_els:
#                 if "Image" in str(type(el)):
#                     images_b64.append(el.metadata.image_base64)
#     return images_b64

# images = get_images_base64(chunks)

# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# # Prompt
# prompt_text = """
# You are an assistant tasked with summarizing tables and text.
# Give a concise summary of the table or text.

# Respond only with the summary, no additionnal comment.
# Do not start your message by saying "Here is a summary" or anything like that.
# Just give the summary as it is.

# Table or text chunk: {element}

# """
# prompt = ChatPromptTemplate.from_template(prompt_text)

# # Summary chain
# model = ChatGroq(temperature=0.5, model="llama-3.1-8b-instant")
# summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

# # Summarize text
# text_summaries = summarize_chain.batch(texts, {"max_concurrency": 3})

# # Summarize tables
# tables_html = [table.metadata.text_as_html for table in tables]
# table_summaries = summarize_chain.batch(tables_html, {"max_concurrency": 3})

# from langchain_openai import ChatOpenAI

# prompt_template = """Describe the image in detail. For context,
#                   the image is part of a research paper explaining the transformers
#                   architecture. Be specific about graphs, such as bar plots."""
# messages = [
#     (
#         "user",
#         [
#             {"type": "text", "text": prompt_template},
#             {
#                 "type": "image_url",
#                 "image_url": {"url": "data:image/jpeg;base64,{image}"},
#             },
#         ],
#     )
# ]

# prompt = ChatPromptTemplate.from_messages(messages)

# chain = prompt | ChatOpenAI(model="gpt-4o-mini") | StrOutputParser()


# image_summaries = chain.batch(images)

# import uuid
# from langchain.vectorstores import Chroma
# from langchain.storage import InMemoryStore
# from langchain.schema.document import Document
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.retrievers.multi_vector import MultiVectorRetriever

# # The vectorstore to use to index the child chunks
# vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=OpenAIEmbeddings())

# # The storage layer for the parent documents
# store = InMemoryStore()
# id_key = "doc_id"

# # The retriever (empty to start)
# retriever = MultiVectorRetriever(
#     vectorstore=vectorstore,
#     docstore=store,
#     id_key=id_key,
# )

# # Add texts
# doc_ids = [str(uuid.uuid4()) for _ in texts]
# summary_texts = [
#     Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
# ]
# retriever.vectorstore.add_documents(summary_texts)
# retriever.docstore.mset(list(zip(doc_ids, texts)))

# # Add tables
# table_ids = [str(uuid.uuid4()) for _ in tables]
# summary_tables = [
#     Document(page_content=summary, metadata={id_key: table_ids[i]}) for i, summary in enumerate(table_summaries)
# ]
# retriever.vectorstore.add_documents(summary_tables)
# retriever.docstore.mset(list(zip(table_ids, tables)))

# # Add image summaries
# img_ids = [str(uuid.uuid4()) for _ in images]
# summary_img = [
#     Document(page_content=summary, metadata={id_key: img_ids[i]}) for i, summary in enumerate(image_summaries)
# ]
# retriever.vectorstore.add_documents(summary_img)
# retriever.docstore.mset(list(zip(img_ids, images)))

# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain_core.messages import SystemMessage, HumanMessage
# from langchain_openai import ChatOpenAI
# from base64 import b64decode


# def parse_docs(docs):
#     """Split base64-encoded images and texts"""
#     b64 = []
#     text = []
#     for doc in docs:
#         try:
#             b64decode(doc)
#             b64.append(doc)
#         except Exception as e:
#             text.append(doc)
#     return {"images": b64, "texts": text}


# def build_prompt(kwargs):

#     docs_by_type = kwargs["context"]
#     user_question = kwargs["question"]

#     context_text = ""
#     if len(docs_by_type["texts"]) > 0:
#         for text_element in docs_by_type["texts"]:
#             context_text += text_element.text

#     # construct prompt with context (including images)
#     prompt_template = f"""
#     Answer the question based only on the following context, which can include text, tables, and the below image.
#     Context: {context_text}
#     Question: {user_question}
#     """

#     prompt_content = [{"type": "text", "text": prompt_template}]

#     if len(docs_by_type["images"]) > 0:
#         for image in docs_by_type["images"]:
#             prompt_content.append(
#                 {
#                     "type": "image_url",
#                     "image_url": {"url": f"data:image/jpeg;base64,{image}"},
#                 }
#             )

#     return ChatPromptTemplate.from_messages(
#         [
#             HumanMessage(content=prompt_content),
#         ]
#     )


# chain = (
#     {
#         "context": retriever | RunnableLambda(parse_docs),
#         "question": RunnablePassthrough(),
#     }
#     | RunnableLambda(build_prompt)
#     | ChatOpenAI(model="gpt-4o-mini")
#     | StrOutputParser()
# )

# chain_with_sources = {
#     "context": retriever | RunnableLambda(parse_docs),
#     "question": RunnablePassthrough(),
# } | RunnablePassthrough().assign(
#     response=(
#         RunnableLambda(build_prompt)
#         | ChatOpenAI(model="gpt-4o-mini")
#         | StrOutputParser()
#     )
# )

# response = chain.invoke(
#     "What is the attention mechanism?"
# )

# print(response)

import os
import pathlib
import uuid
from base64 import b64decode, b64encode

import fitz  # PyMuPDF for PDF + image extraction
import streamlit as st

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma
# from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain.prompts import ChatPromptTemplate

import keys  # Ensure this contains: GOOGLE_API_KEY = "your-key-here"

# Set the Gemini API key
os.environ["GOOGLE_API_KEY"] = keys.GOOGLE_API_KEY

VECTORSTORE_DIR = "chroma_vectorstores"


# ------------------------ STEP 1: Extract images from PDF ------------------------
def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            # Convert to base64 format for Gemini image input
            image_b64 = b64encode(image_bytes).decode()
            images.append(image_b64)
    return images


# ------------------------ STEP 2: Create vector store with text and image chunks ------------------------
def create_multimodal_vectorstore(pdf_path, product_name):
    # Set unique collection name and output folder
    vectorstore_path = os.path.join(VECTORSTORE_DIR, product_name.replace(" ", "_"))
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create Chroma vectorstore
    vectorstore = Chroma(
        collection_name=product_name.replace(" ", "_"),
        embedding_function=embeddings,
        persist_directory=vectorstore_path
    )
    docstore = InMemoryStore()

    # Wrap in multi-vector retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id"
    )

    # Load text from PDF
    loader = PyMuPDFLoader(pdf_path)
    pages = loader.load()
    docs = pages

    # Index text chunks with unique IDs
    doc_ids = [str(uuid.uuid4()) for _ in docs]
    summaries = [
        Document(page_content=doc.page_content, metadata={"doc_id": doc_ids[i]})
        for i, doc in enumerate(docs)
    ]
    retriever.vectorstore.add_documents(summaries)
    retriever.docstore.mset(list(zip(doc_ids, docs)))

    # Index images (base64 strings) as documents
    images = extract_images_from_pdf(pdf_path)
    image_ids = [str(uuid.uuid4()) for _ in images]
    image_docs = [
        Document(page_content=image, metadata={"doc_id": image_ids[i]})
        for i, image in enumerate(images)
    ]
    retriever.vectorstore.add_documents(image_docs)
    retriever.docstore.mset(list(zip(image_ids, images)))

    return retriever


# ------------------------ STEP 3: Parse docs into text & image base64 separately ------------------------
def parse_docs(docs):
    b64, text = [], []
    for doc in docs:
        try:
            # Try decoding to check if it's base64 (image)
            b64decode(doc)
            b64.append(doc)
        except:
            text.append(doc)
    return {"images": b64, "texts": text}


# ------------------------ STEP 4: Build Gemini-compatible multimodal prompt ------------------------
def build_prompt(kwargs):
    docs_by_type = kwargs["context"]
    question = kwargs["question"]

    # Start prompt with combined text context
    prompt_text = "Answer the question based on the context below.\n\n"
    if docs_by_type["texts"]:
        prompt_text += "Text:\n" + "\n".join(
            [doc.text if hasattr(doc, "text") else str(doc) for doc in docs_by_type["texts"]]
        ) + "\n"

    # Add initial text block
    prompt_content = [{"type": "text", "text": prompt_text + f"Question: {question}"}]

    # Add each image as base64 payload
    for image in docs_by_type["images"]:
        prompt_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"},
            }
        )

    # Return Gemini-compatible prompt format
    return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])


# ------------------------ STEP 5: Create the QA chain ------------------------
def get_qa_chain(retriever):
    chain = (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)
        | StrOutputParser()
    )
    return chain


# ------------------------ STREAMLIT INTERFACE ------------------------
st.set_page_config(page_title="Multimodal PDF Q&A", layout="centered")
st.title("📄🤖 Multimodal Gemini Q&A")

# Upload PDF
uploaded_file = st.file_uploader("Upload a Product Manual (PDF)", type=["pdf"])
product_name = st.text_input("Enter Product Name")

if uploaded_file and product_name:
    with st.spinner("🔍 Processing and embedding..."):
        # Save uploaded PDF locally for processing
        temp_pdf_path = os.path.join("temp", f"{product_name}.pdf")
        pathlib.Path("temp").mkdir(exist_ok=True)
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.read())

        # Create multimodal vectorstore + retriever
        retriever = create_multimodal_vectorstore(temp_pdf_path, product_name)
        chain = get_qa_chain(retriever)

        st.success(f"Embeddings updated for '{product_name}' ✅")

        # Ask question
        query = st.text_input("Ask a question about the product manual")

        if query:
            with st.spinner("Generating answer..."):
                answer = chain.invoke(query)
                st.markdown(f"**Answer:** {answer}")

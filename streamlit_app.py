# streamlit_app.py
import streamlit as st
import tempfile, os, hashlib

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import streamlit as st

#os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
#embeddings = OpenAIEmbeddings()
from dotenv import load_dotenv
# load all the keys from .env file
load_dotenv()

st.set_page_config(page_title="Resume Chat", layout="wide")

def save_upload_to_tmp(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

@st.cache_resource
def build_vectorstore_from_path(path, file_hash):
    loader = PyMuPDFLoader(file_path=path)
    pages = loader.load()
    docs = [Document(page_content=p.page_content, id=i+1) for i,p in enumerate(pages)]
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # OpenAI key is read from env (Streamlit Secrets will populate env) or from st.secrets
    # Example: export OPENAI_API_KEY or set in Streamlit Secrets
    embeddings = OpenAIEmbeddings()

    vs = Chroma(
        collection_name=f"resume_{file_hash[:8]}",
        persist_directory=f"./knowledge_base/{file_hash}",
        embedding_function=embeddings
    )
    vs.add_documents(chunks)
    return vs

st.title("Resume QA — LangChain + Chroma + OpenAI")

uploaded = st.file_uploader("Upload resume (PDF)", type=["pdf"])
if uploaded:
    file_bytes = uploaded.getbuffer()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    tmp_path = save_upload_to_tmp(uploaded)

    with st.spinner("Indexing resume (first time takes longest)..."):
        vector_store = build_vectorstore_from_path(tmp_path, file_hash)

    query = st.text_input("Ask a question about this resume:")
    if st.button("Search") and query.strip():
        # try common APIs:
        try:
            docs = vector_store.similarity_search(query, k=5)
        except Exception:
            docs = vector_store.search(query, search_type="similarity", k=5)

        if not docs:
            st.write("No similar passages found.")
        else:
            for i, d in enumerate(docs, 1):
                # adjust depending on returned object shape
                content = getattr(d, "page_content", str(d))
                st.markdown(f"**Result {i}** — {content[:800]}")  # show first 800 chars

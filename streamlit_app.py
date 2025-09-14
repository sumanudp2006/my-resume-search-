import streamlit as st
import tempfile, os, hashlib
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# load env (OPENAI_API_KEY from Streamlit Secrets or .env)
load_dotenv()

st.set_page_config(page_title="Resume Chat (FAISS)", layout="wide")

def save_upload_to_tmp(uploaded_file):
    """Save uploaded PDF to a temporary path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

@st.cache_resource
def build_vectorstore_from_path(path, file_hash):
    """Load resume, split into chunks, and build FAISS vectorstore."""
    loader = PyMuPDFLoader(file_path=path)
    pages = loader.load()
    docs = [Document(page_content=p.page_content, id=i+1) for i, p in enumerate(pages)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    persist_dir = f"./knowledge_base/{file_hash}"
    os.makedirs(persist_dir, exist_ok=True)

    vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vs.save_local(persist_dir)
    return vs

@st.cache_resource
def load_vectorstore_if_exists(file_hash, embeddings):
    """Load FAISS index from disk if available."""
    persist_dir = f"./knowledge_base/{file_hash}"
    if os.path.exists(persist_dir):
        return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    return None

st.title("Resume QA — LangChain + FAISS + OpenAI")

uploaded = st.file_uploader("Upload resume (PDF)", type=["pdf"])
if uploaded:
    file_bytes = uploaded.getbuffer()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    tmp_path = save_upload_to_tmp(uploaded)

    embeddings = OpenAIEmbeddings()
    vector_store = load_vectorstore_if_exists(file_hash, embeddings)

    if not vector_store:
        with st.spinner("Indexing resume (first time takes longest)..."):
            vector_store = build_vectorstore_from_path(tmp_path, file_hash)

    query = st.text_input("Ask a question about this resume:")
    if st.button("Search") and query.strip():
        try:
            docs = vector_store.similarity_search(query, k=5)
        except Exception as e:
            st.error(f"Search error: {e}")
            docs = []

        if not docs:
            st.write("No similar passages found.")
        else:
            for i, d in enumerate(docs, 1):
                content = getattr(d, "page_content", str(d))
                st.markdown(f"**Result {i}** — {content[:800]}")

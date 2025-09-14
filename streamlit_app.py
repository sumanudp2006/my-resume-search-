# streamlit_app.py
import os
import tempfile
import hashlib
from dotenv import load_dotenv
import streamlit as st

# Load local .env for local dev
load_dotenv()

# --- Copy Streamlit Secret to expected env var BEFORE importing/using OpenAI clients ---
# Support both secret names: 'open_ai_api' (what you used) and 'OPENAI_API_KEY'
if "open_ai_api" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["open_ai_api"]
elif "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# If you also set OPENAI_API_KEY in local .env or environment, it's already available.

# Now safe to import langchain/OpenAI modules
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

st.set_page_config(page_title="Resume Chat (FAISS)", layout="wide")

def save_upload_to_tmp(uploaded_file):
    """Save uploaded PDF to a temporary path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

# Cached factory for embeddings (no args => safe to cache)
@st.cache_resource
def get_openai_embeddings():
    try:
        return OpenAIEmbeddings()
    except Exception as e:
        # Let the caller show friendly UI; re-raise for logs
        raise RuntimeError(f"Failed to initialize OpenAIEmbeddings: {e}") from e

# Cache building the index; inputs are hashable (path string and file_hash)
@st.cache_resource
def build_vectorstore_from_path(path: str, file_hash: str):
    loader = PyMuPDFLoader(file_path=path)
    pages = loader.load()
    docs = [Document(page_content=p.page_content, id=i+1) for i, p in enumerate(pages)]

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    embeddings = get_openai_embeddings()  # uses cached factory

    persist_dir = f"./knowledge_base/{file_hash}"
    os.makedirs(persist_dir, exist_ok=True)

    vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
    vs.save_local(persist_dir)

    # Return persist_dir string (hashable). We avoid returning large/unhashable objects.
    return persist_dir

# Do NOT decorate this with @st.cache_resource if it would accept non-hashable args.
# This function only takes a hashable string and returns the loaded FAISS object (not cached).
def load_vectorstore_if_exists(file_hash: str):
    persist_dir = f"./knowledge_base/{file_hash}"
    if os.path.exists(persist_dir):
        embeddings = get_openai_embeddings()
        return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    return None

# st.title("Resume QA — LangChain + FAISS + OpenAI")
if uploaded:
    st.title(f"Chat with your uploaded file - {uploaded.name}")
else:
    st.title("Upload a file to start chatting")


# Friendly check for API key
if not os.environ.get("OPENAI_API_KEY"):
    st.error(
        "OPENAI_API_KEY is not set. You added the secret named 'open_ai_api' — I've mirrored it to "
        "OPENAI_API_KEY automatically. If you're still seeing this, add a secret named OPENAI_API_KEY "
        "or open_ai_api with your key in Manage app → Secrets."
    )
    st.stop()

uploaded = st.file_uploader("Upload resume (PDF)", type=["pdf"])
if uploaded:
    file_bytes = uploaded.getbuffer()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    tmp_path = save_upload_to_tmp(uploaded)

    # Load existing index if present
    vector_store = load_vectorstore_if_exists(file_hash)
    if not vector_store:
        with st.spinner("Indexing resume (first time takes longest)..."):
            try:
                persist_dir = build_vectorstore_from_path(tmp_path, file_hash)
                vector_store = load_vectorstore_if_exists(file_hash)
            except Exception as e:
                st.error(f"Indexing failed: {e}")
                st.stop()

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

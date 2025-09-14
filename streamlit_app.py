# streamlit_app.py
import os
import tempfile
import hashlib
import uuid
from dotenv import load_dotenv
import streamlit as st

# Third-party helper for file locking to avoid race conditions
# pip install filelock
from filelock import FileLock, Timeout

# Load local .env for local dev
load_dotenv()

# --- Copy Streamlit Secret to expected env var BEFORE importing/using OpenAI clients ---
if "open_ai_api" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["open_ai_api"]
elif "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Now safe to import langchain/OpenAI modules
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Resume Chat (FAISS)", layout="wide")

# --- Utility: per-session user id so session_state caching is isolated per user ---
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())

USER_ID = st.session_state["user_id"]

def save_upload_to_tmp(uploaded_file):
    """Save uploaded PDF to a unique temporary path for the session."""
    # use session id to keep files separated across concurrent sessions
    tmp_dir = os.path.join(tempfile.gettempdir(), "resume_chat", USER_ID)
    os.makedirs(tmp_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf", dir=tmp_dir)
    os.close(fd)
    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return tmp_path

# Cached factory for embeddings (no args => safe to cache across sessions)
@st.cache_resource
def get_openai_embeddings():
    try:
        return OpenAIEmbeddings()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize OpenAIEmbeddings: {e}") from e

# Build vectorstore with a file lock to avoid concurrent writes
@st.cache_resource
def build_vectorstore_from_path(path: str, file_hash: str, source_name: str, persist_root: str = "./knowledge_base"):
    """
    Build FAISS vectors from the PDF at `path` and save in persist_root/<file_hash>.
    This function is cached by Streamlit (so identical args bypass re-run).
    We use a file lock while creating/saving the index to avoid corrupting it in concurrent runs.
    """
    persist_dir = os.path.join(persist_root, file_hash)
    lock_path = persist_dir + ".lock"
    lock = FileLock(lock_path)

    # If directories already exist and appear valid, load and return
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        # We'll still return the persist_dir string; caller loads it with embeddings
        return persist_dir

    # Acquire lock - wait up to 30s then fail (adjust in prod)
    try:
        with lock.acquire(timeout=30):
            # Re-check after acquiring lock (another process may have finished)
            if os.path.exists(persist_dir) and os.listdir(persist_dir):
                return persist_dir

            loader = PyMuPDFLoader(file_path=path)
            pages = loader.load()

            # create Documents with page metadata for provenance (kept internally)
            docs = [
                Document(page_content=p.page_content, metadata={"page": i + 1, "source": source_name})
                for i, p in enumerate(pages)
            ]

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(docs)

            embeddings = get_openai_embeddings()
            os.makedirs(persist_dir, exist_ok=True)

            vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
            # Save index atomically by writing into a temp dir and renaming (FAISS.save_local writes files)
            vs.save_local(persist_dir)

            return persist_dir
    except Timeout:
        # Lock couldn't be acquired - try to proceed by returning persist_dir if exists
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            return persist_dir
        raise RuntimeError("Index is being built by another process and timed out waiting for lock.")

def load_vectorstore_if_exists(file_hash: str, persist_root: str = "./knowledge_base"):
    persist_dir = os.path.join(persist_root, file_hash)
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        embeddings = get_openai_embeddings()
        # load_local may be heavy; we keep it per-session in st.session_state below
        return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
    return None

# Friendly check for API key
if not os.environ.get("OPENAI_API_KEY"):
    st.error(
        "OPENAI_API_KEY is not set. Add a secret named OPENAI_API_KEY or open_ai_api with your key in Manage app → Secrets."
    )
    st.stop()

# UI
uploaded = st.file_uploader("Upload resume (PDF)", type=["pdf"])
if uploaded:
    st.title(f"Chat with your uploaded file - {uploaded.name}")
else:
    st.title("Upload a file to start chatting")

# Keep vector_store loaded per session to avoid reloading for every interaction
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None
if "persist_hash" not in st.session_state:
    st.session_state["persist_hash"] = None

if uploaded:
    file_bytes = uploaded.getbuffer()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    tmp_path = save_upload_to_tmp(uploaded)

    # If session already has the same index, reuse it
    if st.session_state["persist_hash"] == file_hash and st.session_state["vector_store"] is not None:
        vector_store = st.session_state["vector_store"]
    else:
        # Attempt to load existing index (fast path)
        vector_store = None
        try:
            vector_store = load_vectorstore_if_exists(file_hash)
        except Exception as e:
            # Loading may fail if index being written; we proceed to build with lock below
            st.warning(f"Index load encountered an issue: {e}")

        # If index not available, build it (with lock inside build function)
        if not vector_store:
            with st.spinner("Indexing resume (first time takes longest)..."):
                try:
                    persist_dir = build_vectorstore_from_path(tmp_path, file_hash, source_name=uploaded.name)
                    vector_store = load_vectorstore_if_exists(file_hash)
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
                    st.stop()

        # Save vector_store into session so subsequent calls in this session reuse it
        st.session_state["vector_store"] = vector_store
        st.session_state["persist_hash"] = file_hash

    # Strict template to avoid fabrication
    template = """
You are a helpful assistant answering all the user questions.
Answer the user questions based on the context provided.
If you do not know the answer, please say "Don't know". 
Do not fabricate the answer at any cost.

Question: {question}
Context: {context}
"""
    prompt = PromptTemplate(template=template, input_variables=["question", "context"])

    query = st.text_input("Ask a question about this resume:")
    if st.button("Search") and query.strip():
        if not st.session_state["vector_store"]:
            st.error("Vector store not loaded.")
        else:
            try:
                # Build a retriever (use top k results)
                # NOTE: keeping k modest for production concurrency and token limits
                retriever = st.session_state["vector_store"].as_retriever(search_type="similarity", search_kwargs={"k": 5})

                # Initialize deterministic LLM
                llm = ChatOpenAI(model="gpt-5-nano", temperature=0.0, max_tokens=1024)

                # RetrievalQA chain uses prompt; keep return_source_documents=True so chain has context,
                # but we intentionally do not display sources in the UI.
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt},
                )

                # Execute - this runs in the current Streamlit session/thread
                with st.spinner("Generating answer..."):
                    result = qa({"query": query})
                answer = result.get("result") or result.get("answer") or ""
                answer = answer.strip()

                st.markdown("### Answer")
                st.write(answer if answer else "Don't know")

            except Exception as e:
                st.error(f"Search / QA failed: {e}")

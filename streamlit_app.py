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
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

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
def build_vectorstore_from_path(path: str, file_hash: str, source_name: str):
    loader = PyMuPDFLoader(file_path=path)
    pages = loader.load()

    # Create Documents with metadata (page numbers + source filename) for provenance
    docs = [
        Document(page_content=p.page_content, metadata={"page": i + 1, "source": source_name})
        for i, p in enumerate(pages)
    ]

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
    st.title(f"Chat with your uploaded file - {uploaded.name}")
else:
    st.title("Upload a file to start chatting")

if uploaded:
    file_bytes = uploaded.getbuffer()
    file_hash = hashlib.md5(file_bytes).hexdigest()
    tmp_path = save_upload_to_tmp(uploaded)

    # Load existing index if present
    vector_store = load_vectorstore_if_exists(file_hash)
    if not vector_store:
        with st.spinner("Indexing resume (first time takes longest)..."):
            try:
                persist_dir = build_vectorstore_from_path(tmp_path, file_hash, source_name=uploaded.name)
                vector_store = load_vectorstore_if_exists(file_hash)
            except Exception as e:
                st.error(f"Indexing failed: {e}")
                st.stop()

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
        if not vector_store:
            st.error("Vector store not loaded.")
        else:
            try:
                # Build a retriever (use top k results)
                retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

                # Initialize the LLM with deterministic settings to minimize hallucination
                # NOTE: If gpt-5-nano isn't available in your environment/account, change model here.
                llm = ChatOpenAI(model="gpt-5-nano", temperature=0.0, max_tokens=1024)

                # RetrievalQA chain using the custom prompt. return_source_documents=True to see provenance.
                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",              # "stuff" is simplest: it stuffs context into prompt
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt},
                )

                # Run the chain
                result = qa({"query": query})
                # langchain variations return "result" or "answer"
                answer = result.get("result") or result.get("answer") or ""
                source_docs = result.get("source_documents", [])

                # Show the answer
                st.markdown("### Answer")
                st.write(answer.strip())

                # Show the sources (short excerpts + metadata)
                if source_docs:
                    st.markdown("#### Source passages (top results)")
                    for i, src in enumerate(source_docs, 1):
                        content = getattr(src, "page_content", "")
                        metadata = getattr(src, "metadata", {}) or {}
                        page = metadata.get("page", "unknown")
                        source = metadata.get("source", "unknown")
                        st.markdown(f"**Source {i} — {source} (page {page})**")
                        st.text(content[:800].strip())
                        if metadata:
                            st.caption(f"metadata: {metadata}")
                else:
                    st.write("No supporting passages returned by retriever.")
            except Exception as e:
                st.error(f"Search / QA failed: {e}")

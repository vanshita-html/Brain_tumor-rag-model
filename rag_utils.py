import os
from dotenv import load_dotenv
import glob
import fitz  # PyMuPDF
import pickle
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Try importing google.generativeai
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_STORE_DIR = "vector_store"
INDEX_FILE_PATH = os.path.join(VECTOR_STORE_DIR, "vector_index.pkl")
KNOWLEDGE_BASE_DIR = "knowledge_base"
load_dotenv()

DEFAULT_USER_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
# Candidate Gemini model names in order of recommendation
MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-3.6-flash",
    "gemini-flash-latest",
    "gemini-1.5-flash",
    "gemini-2.0-flash",
    "gemini-3.5-flash",
    "gemini-pro-latest"
]

@st.cache_resource
def get_embedding_model():
    """Load and cache local SentenceTransformer embedding model."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

def chunk_text(text, chunk_size=750, chunk_overlap=150):
    """
    Split long document text into overlapping chunks using RecursiveCharacterTextSplitter
    so sentences and words are never cut in half.
    ~750 characters long with ~150 characters overlap.
    """
    if not text or not text.strip():
        return []
        
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

def extract_pdf_chunks(pdf_path, chunk_size=750, chunk_overlap=150):
    """Extract text from PDF page by page and generate chunk dicts with metadata."""
    doc = fitz.open(pdf_path)
    doc_name = os.path.basename(pdf_path)
    doc_chunks = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text("text")
        if not page_text.strip():
            continue
            
        page_chunks = chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for c in page_chunks:
            doc_chunks.append({
                'doc_name': doc_name,
                'page': page_num + 1,
                'text': c
            })
            
    doc.close()
    return doc_chunks

def build_vector_index(kb_dir=KNOWLEDGE_BASE_DIR, save_dir=VECTOR_STORE_DIR):
    """
    Parse all PDFs in knowledge_base, compute embeddings using SentenceTransformer,
    and persist vector index to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    pdf_files = glob.glob(os.path.join(kb_dir, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in '{kb_dir}' to build index.")
        return None
        
    all_chunks = []
    for pdf_file in pdf_files:
        chunks = extract_pdf_chunks(pdf_file)
        all_chunks.extend(chunks)
        
    if not all_chunks:
        print("No text content could be extracted from PDFs.")
        return None
        
    embedder = get_embedding_model()
    texts = [c['text'] for c in all_chunks]
    
    # Compute embeddings (normalized for cosine similarity)
    embeddings = embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype=np.float32)
    
    index_data = {
        'embeddings': embeddings,
        'chunks': all_chunks,
        'model_name': EMBEDDING_MODEL_NAME
    }
    
    index_path = os.path.join(save_dir, "vector_index.pkl")
    with open(index_path, "wb") as f:
        pickle.dump(index_data, f)
        
    print(f"Successfully indexed {len(all_chunks)} chunks from {len(pdf_files)} PDFs -> {index_path}")
    return index_data

@st.cache_resource
def load_vector_store():
    """
    Load persisted vector store from disk.
    If vector store does not exist yet, build it automatically.
    """
    if not os.path.exists(INDEX_FILE_PATH):
        index_data = build_vector_index()
        if index_data is None:
            return None
        return index_data
        
    with open(INDEX_FILE_PATH, "rb") as f:
        index_data = pickle.load(f)
        
    return index_data

def query_vector_store(query, top_k=4):
    """
    Retrieve top_k relevant chunks from persisted vector store using cosine similarity.
    """
    index_data = load_vector_store()
    if not index_data:
        return []
        
    embedder = get_embedding_model()
    query_emb = embedder.encode([query], normalize_embeddings=True)[0]
    
    embeddings = index_data['embeddings']
    chunks = index_data['chunks']
    
    # Cosine similarities
    scores = np.dot(embeddings, query_emb)
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            'score': float(scores[idx]),
            'doc_name': chunks[idx]['doc_name'],
            'page': chunks[idx]['page'],
            'text': chunks[idx]['text']
        })
        
    return results

def format_prompt(query, retrieved_chunks, chat_history=None):
    """Construct grounded prompt for Gemini LLM."""
    context_str = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_str += f"\n--- Document [{i}]: {chunk['doc_name']} (Page {chunk['page']}) ---\n{chunk['text']}\n"
        
    history_str = ""
    if chat_history:
        recent_turns = chat_history[-6:]  # Last 3 Q&A pairs
        for msg in recent_turns:
            role = "Patient" if msg['role'] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"
            
    prompt = f"""System Instructions:
You are an empathetic, knowledgeable medical research assistant designed for brain tumor inquiries.
Follow these strict rules:
1. Answer the patient's question using ONLY the provided reference documents context.
2. Synthesize a clear, coherent paragraph in plain, reassuring, non-alarming language suitable for patients.
3. Do NOT dump raw excerpt quotes with page numbers inside the main answer body.
4. If the answer is not present in the reference context, state clearly: "I could not find specific information on this in the reference medical guides."
5. Always end responses related to symptoms, diagnosis, or treatment with a warm reminder to consult a qualified medical doctor for personalized clinical evaluation.

Reference Context Documents:
{context_str}

Recent Chat History:
{history_str}

Patient Question:
{query}

Assistant Response:"""
    return prompt

def get_gemini_key():
    """Get Gemini API key from environment variable or fallback user key."""
    load_dotenv()
    env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    return DEFAULT_USER_KEY

def ask_rag_assistant(query, chat_history=None):
    """
    Execute full RAG pipeline:
    1. Retrieve top-k chunks
    2. Call Gemini API if GEMINI_API_KEY is present
    3. Return synthesized response or explicit error message
    """
    retrieved_chunks = query_vector_store(query, top_k=4)
    
    if not retrieved_chunks:
        return {
            'answer': "I couldn't locate any matching reference documents in the knowledge base. Please try rephrasing your question.",
            'sources': [],
            'debug': "No chunks found in vector store"
        }
        
    api_key = get_gemini_key()
    
    if GENAI_AVAILABLE and api_key:
        try:
            genai.configure(api_key=api_key)
            last_error = None
            response_text = None
            used_model = None
            
            for mname in MODEL_CANDIDATES:
                try:
                    model = genai.GenerativeModel(mname)
                    prompt = format_prompt(query, retrieved_chunks, chat_history)
                    response = model.generate_content(prompt)
                    if response and response.text:
                        response_text = response.text
                        used_model = mname
                        break
                except Exception as err:
                    last_error = str(err)
                    continue
                    
            if response_text:
                debug_msg = f"Gemini Connected ({used_model})"
                return {
                    'answer': response_text,
                    'sources': retrieved_chunks,
                    'debug': debug_msg
                }
            else:
                error_detail = f"Gemini API Call Failed: {last_error}"
                st.error(f"⚠️ {error_detail}")
                return {
                    'answer': f"⚠️ **Gemini API Error**: {last_error}\n\n*Displaying retrieved reference context below as backup:*",
                    'sources': retrieved_chunks,
                    'debug': error_detail
                }
        except Exception as e:
            error_detail = f"Gemini Config Error: {str(e)}"
            st.error(f"⚠️ {error_detail}")
            
    # Fallback if no key or genai uninstalled
    fallback_text = (
        "💡 **Reference Knowledge Base Passages** *(Gemini API key not configured — displaying retrieved medical references directly)*:\n\n"
    )
    for i, c in enumerate(retrieved_chunks, 1):
        fallback_text += f"**Excerpt {i}** *(from {c['doc_name']}, Page {c['page']})*:\n\"{c['text']}\"\n\n"
        
    fallback_text += (
        "\n---  \n"
        "🏥 **Medical Note**: Always consult a qualified doctor or neuro-oncologist for professional diagnostic evaluations."
    )
    
    return {
        'answer': fallback_text,
        'sources': retrieved_chunks,
        'debug': "API key not set"
    }

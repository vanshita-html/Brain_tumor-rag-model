import sys
import os
from rag_utils import build_vector_index, KNOWLEDGE_BASE_DIR, VECTOR_STORE_DIR

def main():
    print("=" * 60)
    print("Brain Tumor RAG Assistant — Vector Index Builder")
    print("=" * 60)
    print(f"Scanning knowledge base directory: '{KNOWLEDGE_BASE_DIR}'")
    
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        print(f"Error: Directory '{KNOWLEDGE_BASE_DIR}' does not exist.")
        sys.exit(1)
        
    index_data = build_vector_index(kb_dir=KNOWLEDGE_BASE_DIR, save_dir=VECTOR_STORE_DIR)
    
    if index_data:
        num_chunks = len(index_data['chunks'])
        print(f"\n[SUCCESS] Vector store index successfully created with {num_chunks} chunks!")
        print(f"Persisted index file: '{os.path.join(VECTOR_STORE_DIR, 'vector_index.pkl')}'")
    else:
        print("\n[WARNING] Failed to build vector index. Please check your PDF files.")

if __name__ == "__main__":
    main()

"""
goal of script: to convert pdf's into processable data for the app (vector database)
"""

from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# use directoryloader w unstructured as the underlying loader
def load_pdf_documents():
    """Load PDF documents from a directory and convert them into processable data."""
    loader = DirectoryLoader(
    "./data", 
    glob="**/*.pdf", 
    loader_cls=UnstructuredPDFLoader, 
    loader_kwargs={"mode": "elements"}, 
    show_progress=True
    )
    docs = loader.load()
    return docs
# 10-k is too long to process the way it is, need to split loaded docs into smaller chunnks (1000 char is a good start)
def split_documents(docs):
    """Split documents into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200, 
        length_function=len
    )
    return text_splitter.split_documents(docs)

def embed_documents(chunkedDocs):
    """Embed documents using a vector store.""" 
    texts = [doc.page_content for doc in chunkedDocs]
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=True)
    # Convert embeddings to a list of lists for compatibility with ChromaDB
    embeddings = embeddings.tolist()
    return embeddings

def store_embeddings(chunkedDocs, embeddings, batch_size=4000):
    """Store embeddings and documents in ChromaDB with batch processing.
    """
    chroma_client = chromadb.PersistentClient()
    collection = chroma_client.get_or_create_collection(name="chunkedTextEmbeddings")
    
    # Process in batches cuz of chromadb limit
    for i in range(0, len(chunkedDocs), batch_size):
        batch_docs = chunkedDocs[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        
        documents = [doc.page_content for doc in batch_docs]
        metadatas = [{"source": doc.metadata.get("source", "unknown")} for doc in batch_docs]
        ids = [f"doc_{i+j}" for j in range(len(batch_docs))]
        
        collection.add(
            documents=documents,
            embeddings=batch_embeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Processed batch {i//batch_size + 1}/{(len(chunkedDocs)-1)//batch_size + 1}")
        
def ingestData():
    """Main function to ingest data from PDFs."""
    docs = load_pdf_documents()
    print(f"Loaded {len(docs)} documents.")
    
    split_docs = split_documents(docs)
    print(f"Split into {len(split_docs)} chunks.")
    
    embeddings = embed_documents(split_docs)
    print(f"Generated embeddings for {len(embeddings)} chunks.")
    
    store_embeddings(split_docs, embeddings)
    print("Embeddings stored in the vector database.")
    
if __name__ == "__main__": # Run the ingestion process
    ingestData()

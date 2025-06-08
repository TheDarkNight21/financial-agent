"""
goal of script: to convert pdf's into processable data for the app (vector database)
"""

from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# use directoryloader w unstructured as the underlying loader
def loadPdfDocuments():
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
def splitDocuments(docs):
    """Split documents into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],  # Prioritize splitting on logical breaks
        length_function=len,
        add_start_index=True  # Helpful for tracking where the chunk came from
    )
    return text_splitter.split_documents(docs)

def embedDocuments(chunkedDocs):
    """Embed documents using a vector store.""" 
    texts = [doc.page_content for doc in chunkedDocs]
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=True)
    # Convert embeddings to a list of lists for compatibility with ChromaDB
    embeddings = embeddings.tolist()
    return embeddings

def storeEmbeddings(chunkedDocs, embeddings, batch_size=4000):
    """Store embeddings and documents in ChromaDB with batch processing.
    """
    chromaClient = chromadb.PersistentClient()
    collection = chromaClient.get_or_create_collection(name="chunkedTextEmbeddings")
    
    # Process in batches cuz of chromadb limit
    for i in range(0, len(chunkedDocs), batch_size):
        batchDocs = chunkedDocs[i:i+batch_size]
        batchEmbeddings = embeddings[i:i+batch_size]
        
        documents = [doc.page_content for doc in batchDocs]
        metadatas = [{"source": doc.metadata.get("source", "unknown")} for doc in batchDocs]
        ids = [f"doc_{i+j}" for j in range(len(batchDocs))]
        
        collection.add(
            documents=documents,
            embeddings=batchEmbeddings,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Processed batch {i//batch_size + 1}/{(len(chunkedDocs)-1)//batch_size + 1}")
        
def ingestData():
    """Main function to ingest data from PDFs."""
    docs = loadPdfDocuments()
    print(f"Loaded {len(docs)} documents.")
    
    splitDocs = splitDocuments(docs)
    print(f"Split into {len(splitDocs)} chunks.")
    
    embeddings = embedDocuments(splitDocs)
    print(f"Generated embeddings for {len(embeddings)} chunks.")
    
    storeEmbeddings(splitDocs, embeddings)
    print("Embeddings stored in the vector database.")
    
if __name__ == "__main__": # run it
    ingestData()

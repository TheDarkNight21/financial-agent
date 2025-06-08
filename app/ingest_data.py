"""
Goal of script: to convert PDFs into processable data for the app (vector database)
"""

# --- Imports ---
# Document loading and splitting
from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Custom embeddings and vectorstore
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma

# --- Custom Embedding Class ---
# This class is well-written and correctly integrates SentenceTransformers with LangChain.
class CustomSentenceTransformerEmbeddings(Embeddings):
    """
    A custom LangChain embedding class for SentenceTransformer models.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        # The model.encode function is highly optimized for batch processing.
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text).tolist()


# --- Data Loading and Processing Functions ---
def loadPdfDocuments(directory_path: str = "./data"):
    """
    Loads all PDF documents from a specified directory using UnstructuredPDFLoader.
    Using mode="elements" is good for capturing structure like titles and lists.
    """
    print(f"Loading documents from {directory_path}...")
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=UnstructuredPDFLoader,
        loader_kwargs={"mode": "elements"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} document elements.")
    return docs


def splitDocuments(docs):
    """
    Splits the loaded documents into smaller, more manageable chunks.
    This is crucial for effective embedding and retrieval.
    """
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],  # Sensible defaults
        length_function=len,
        add_start_index=True,  # Helpful for source tracking
    )
    split_docs = text_splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} chunks.")
    return split_docs


def filter_metadata(metadata: dict) -> dict:
    """
    Filters metadata to ensure all values are of a simple, storable type.
    This prevents errors with some vector database backends.
    """
    simple_meta = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            simple_meta[k] = v
    return simple_meta


# --- Main Ingestion Logic ---
def ingestData(persist_directory: str = "./chroma_db"):
    """
    Main function to load, split, embed, and store PDF data in a persistent Chroma vectorstore.
    This revised version uses the more efficient `Chroma.from_documents` method.
    """
    # 1. Load and split the documents
    docs = loadPdfDocuments()
    split_docs = splitDocuments(docs)

    # 2. Filter metadata for compatibility (a good practice)
    for doc in split_docs:
        doc.metadata = filter_metadata(doc.metadata)

    # 3. Initialize the custom embedding model
    embedding_model = CustomSentenceTransformerEmbeddings()

    # 4. Create and persist the vectorstore in a single, optimized step
    # `Chroma.from_documents` handles the entire process:
    #   - It takes the documents and the embedding function.
    #   - It automatically handles creating embeddings for each document (in batches).
    #   - It initializes the Chroma vectorstore.
    #   - It adds the documents with their embeddings and metadata.
    #   - It saves (persists) the final database to the specified directory.
    print("Creating and persisting vectorstore...")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name="pdf_embeddings" # Using a descriptive collection name
    )

    print(f"Successfully created and persisted vectorstore to '{persist_directory}'.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # This will run the full ingestion pipeline when the script is executed.
    # Make sure you have a './data' directory with some PDFs in it.
    ingestData(persist_directory="./chroma_db")


# --- HOW TO LOAD AND QUERY THE DATABASE IN YOUR APP ---
#
# The most common reason for getting "nothing" back is failing to load the
# persisted database correctly. You must initialize Chroma with the same
# persist_directory and embedding function.
#
# Here is an example of what your query function should look like in your
# main application (e.g., app.py, api.py).

def query_database(query_text: str, persist_directory: str = "./chroma_db"):
    """
    Loads the persisted vector database and performs a similarity search.
    """
    print("Loading existing vectorstore...")

    # 1. Initialize the same embedding model used during ingestion
    embedding_model = CustomSentenceTransformerEmbeddings()

    # 2. Load the persisted database from disk
    #    THIS IS THE CRUCIAL STEP. By providing the `persist_directory`,
    #    Chroma knows to load the data from that folder. If you omit this,
    #    it will create a new, EMPTY database in memory.
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name="pdf_embeddings"
    )

    print(f"Vectorstore loaded. Searching for: '{query_text}'")

    # 3. Perform a similarity search
    results = vectorstore.similarity_search(query_text, k=4) # Find the 4 most similar documents

    if not results:
        print("No results found.")
        return

    print(f"\nFound {len(results)} relevant chunks:\n")
    for i, doc in enumerate(results):
        print(f"--- Result {i+1} ---")
        print(f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page_number', 'N/A')}")
        print(f"Content: {doc.page_content[:250]}...")
        print("-" * 20)


# Example of how to use the query function.
# You would call this from your application logic.
if __name__ == "__main__":
    print("\n" + "="*50)
    print("RUNNING A TEST QUERY")
    print("="*50)
    # You would replace this with a real query from your application
    test_query = "what is the main topic of the documents?"
    query_database(test_query)
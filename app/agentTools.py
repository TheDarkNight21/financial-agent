import langchain_core
import yfinance as yf
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL
from langchain_chroma import Chroma

# --- CHANGE #1: Use pathlib for robust path handling ---
# This creates a reliable, absolute path to your project's root directory
# and then to the chroma folder within it.
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIRECTORY = PROJECT_ROOT / "chroma"

# --- CHANGE #2: Import the shared embedding class from a 'core' directory ---
# This assumes you have moved the CustomSentenceTransformerEmbeddings class
# into a file like /core/embeddings.py for cleaner organization.
from app.ingestData import CustomSentenceTransformerEmbeddings


# --- RAG Tool for SEC Filings (Corrected) ---
def ragTool():
    """
    Search through 10-K files from a ChromaDB vectorstore and return relevant information.
    """
    print("Initializing RAG tool...")

    # Check if the database directory actually exists to give a clear error.
    if not PERSIST_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Vectorstore not found at '{PERSIST_DIRECTORY}'. "
            f"Please ensure you have run ingest_data.py and the database was created in the project root."
        )
    
    embedding_model = CustomSentenceTransformerEmbeddings()

    # --- CHANGE #3: Use the robust, absolute path to load the database ---
    # We convert the Path object to a string, which is what the function expects.
    vectorstore = Chroma(
        persist_directory=str(PERSIST_DIRECTORY),  
        embedding_function=embedding_model,
        collection_name="pdf_embeddings" 
    )
    
    doc_count = vectorstore._collection.count()
    print(f"Successfully loaded vectorstore from '{PERSIST_DIRECTORY}'. Number of documents: {doc_count}")

    if doc_count == 0:
        print("Warning: Vectorstore is empty. The RAG tool will not find any documents.")

    baseRetriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    retrieverTool = langchain_core.tools.retriever.create_retriever_tool(
        retriever=baseRetriever,
        name="sec_filing_search",
        description="""Use this to answer questions about a company's financial performance, business strategy, 
        risks, and executive commentary by searching through their 10-K SEC filings. 
        For example: 'What were the main business risks for Apple in their latest 10-K?'"""
    )

    print("RAG tool created successfully.")
    return retrieverTool


# --- Other Tools (No changes needed) ---

def getStockPrice(ticker: str) -> str:
    """
    Fetches the latest stock price for a given ticker symbol using yfinance.
    """
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d", interval="1m")['Close'].iloc[-1]
        return f"The current price of {ticker.upper()} is ${price:.2f}."
    except Exception as e:
        return f"Could not retrieve price for {ticker.upper()}: {e}"

def stockPriceTool():
    """
    Creates a LangChain Tool for getting the current stock price.
    """
    stockTool = Tool(
        name="stock_price_lookup",
        func=getStockPrice,
        description="Useful for getting the current stock price for a given stock ticker symbol. For example, 'MSFT' for Microsoft or 'AAPL' for Apple."
    )
    return stockTool

def calculatorTool():
    """
    Creates a LangChain Tool that uses a Python REPL for calculations.
    """
    pythonRepl = PythonREPL()
    repl_tool = Tool(
        name="python_repl_calculator",
        description="""A calculator that executes a line of python code and returns the result. 
        Use this for any mathematical calculations. The code runs in a sandbox.
        Example: '3.14 * 2**2'""",
        func=pythonRepl.run,
    )
    return repl_tool


# --- Verification Step (Updated to use the new path) ---
if __name__ == "__main__":
    # Test 1: Ensure the ragTool() function can be called without errors and creates a Tool.
    print("--- Testing RAG Tool Initialization ---")
    rag_tool_instance = ragTool()
    print(f"Successfully created a tool named '{rag_tool_instance.name}' of type {type(rag_tool_instance)}")
    
    # Test 2: Directly test the retriever logic to ensure it pulls documents.
    # This isolates the test to just the retrieval part.
    print("\n--- Testing direct retrieval with a sample query ---")
    try:
        # Re-create the core components needed for retrieval using the CORRECT path
        embedding_model = CustomSentenceTransformerEmbeddings()
        
        # --- CHANGE #4: Update the test to use the same robust path as the tool ---
        vectorstore = Chroma(
            persist_directory=str(PERSIST_DIRECTORY),
            embedding_function=embedding_model,
            collection_name="pdf_embeddings"
        )
        
        retriever_for_test = vectorstore.as_retriever()

        test_query = "What risks did Meta mention?"
        print(f"Directly invoking retriever with query: '{test_query}'")
        
        results = retriever_for_test.invoke(test_query)
        
        if results:
            print(f"\nFound {len(results)} results:")
            for i, doc in enumerate(results):
                print(f"\n--- Result {i+1} ---")
                print(f"Source: {doc.metadata.get('source', 'N/A')}")
                print(f"Content: {doc.page_content[:300]}...")
        else:
            print("No results found from direct retrieval test. Check your DB and query.")

    except Exception as e:
        print(f"\nAn error occurred during the direct retrieval test: {e}")
        print(f"Please ensure '{PERSIST_DIRECTORY}' directory exists and contains data.")
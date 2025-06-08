import chromadb
import langchain_core
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

import yfinance as yf
from langchain.agents import Tool

from langchain_experimental.utilities import PythonREPL



def ragTool():
    """
    search thru 10-k files from chroma db and return relevant information
    """
    vectorStore = chromadb.load(persist_directory=".")
    baseRetriever = vectorStore.as_retriever()
    
    retrieverTool = langchain_core.tools.retriever.create_retriever_tool(
        retriever = baseRetriever,
        name = "sec_filing_search",
        description = """Useful for answering questions about a company's financial performance, business strategy, 
        risks, and executive commentary. Use this to find information from official 10-K SEC filings.""",
    )

    return retrieverTool

def getStockPrice(ticker: str) -> str:
    """
    simple get stock price fetcher function for tool using ticker.
    """
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")["Close"].iloc[-1]
        return f"The current price of {ticker.upper()} is ${price:.2f}."
    except Exception as e:
        return f"Could not retrieve price for {ticker.upper()}: {e}"


def stockPriceTool():
    """
    uses get stock price function to get current stock price of a given ticker symbol that can be used by agent.
    """
    stockTool = Tool(
        name="stock_price_lookup",
        func=getStockPrice,
        description="Useful for getting the current stock price for a given stock ticker symbol. For example, 'MSFT' for Microsoft."
    )
    
    return stockTool

def calculatorTool():
    """
    uses python repl to create a calculator tool that can be used by agent.
    """
    pythonRepl = PythonREPL()
    repl_tool = Tool(
        name="python_repl",
        description="""Executes python code and returns the result. The code runs in a static sandbox without interactive mode, 
        so print output or save output to a file.""",
        func=pythonRepl.run,
    )
    
    return repl_tool



from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from langchain_openai import ChatOpenAI
# THIS IS THE CORRECT IMPORT
from .agentTools import ragTool, stockPriceTool, calculatorTool 
from .ingestData import CustomSentenceTransformerEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()
openaiKey = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# It's good practice to initialize these objects once when the app starts
memory = MemorySaver()
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=openaiKey)
tools = [
    ragTool(), 
    stockPriceTool(), 
    calculatorTool()
]
agentExecutor = create_react_agent(llm, tools, checkpointer=memory)

class Query(BaseModel): 
    question: str
    
@app.post("/ask")
async def ask_question(query: Query):
    try:
        # Note: LangGraph's .invoke expects a dictionary with specific keys.
        # The key for the user's question is usually 'input'.
        # The config is for managing conversational state.
        config = {"configurable": {"thread_id": "some-thread-id"}} # You'll want to manage this per-user/session
        
        # Invoke the agent with the correct input structure
        result = agentExecutor.invoke(
            {"messages": [("human", query.question)]}, 
            config=config
        )
        
        # The final answer is in the last message of the result
        final_response = result['messages'][-1].content
        
        return JSONResponse(content={"response": final_response})
    except Exception as e:
        # It's helpful to log the error for debugging
        print(f"An error occurred: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
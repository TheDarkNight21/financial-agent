from langchain_openai import ChatOpenAI
from agentTools import ragTool, stockPriceTool, calculatorTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
import dotenv

dotenv.load_dotenv()


memory = MemorySaver()
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
tools = [ragTool(), 
         stockPriceTool(), 
         calculatorTool()
         ]
agent_executor = create_react_agent(llm, tools, checkpointer=memory)

# Example user query combining both tools
user_query = "If MSFT stock is trading at $400 and its EPS is $11, what is the P/E ratio?"

# Configuration
config = {"configurable": {"thread_id": "abc123"}}  # optional: for multi-turn memory/identity

# Streaming agent response using your agent_executor (assumed set up with tools + LLM)
for step in agent_executor.stream(
    {"messages": [HumanMessage(content=user_query)]},
    config,
    stream_mode="values",
):
    step["messages"][-1].pretty_print()

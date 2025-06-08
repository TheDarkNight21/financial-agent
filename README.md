# Financial Research AI Agent

This project is a fully containerized, end-to-end financial research agent. The agent is capable of answering complex financial queries by autonomously selecting and using a set of tools, including a RAG pipeline for SEC filings, a live financial data API, and a Python calculator.

The entire application is deployed as a REST API using FastAPI and containerized with Docker for easy setup and portability.

## ✨ Features

*   **Agentic Reasoning:** Uses a Langchain agent to decompose complex questions and perform multi-step reasoning to find answers.
*   **Multi-Tool Integration:** The agent has access to multiple tools it can choose from:
    1.  **SEC Filing Search (RAG):** Performs Retrieval-Augmented Generation on a local vector database of 10-K filings.
    2.  **Live Stock Price API:** Fetches real-time stock prices using the `yfinance` library.
    3.  **Python REPL Calculator:** Executes Python code to perform on-the-fly calculations (e.g., P/E ratios).
*   **REST API Deployment:** The agent is exposed via a robust FastAPI endpoint, making it easy to integrate with other applications.
*   **Containerized & Reproducible:** The entire application and its dependencies are packaged in a Docker container, ensuring it runs the same way everywhere.

## 🛠️ Technology Stack

*   **Backend:** Python
*   **AI Orchestration:** LangChain
*   **API Framework:** FastAPI
*   **Vector Database:** ChromaDB
*   **Containerization:** Docker
*   **LLM Provider:** OpenAI

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   **Docker:** You must have Docker installed and running on your machine. [Download Docker](https://www.docker.com/products/docker-desktop/).
*   **OpenAI API Key:** The agent uses an OpenAI model (like GPT-3.5 or GPT-4) as its brain. You will need an API key from [platform.openai.com](https://platform.openai.com/).

### Installation & Setup

Follow these steps to set up the project environment. All commands should be run from the root directory of this project.

**1. Clone the Repository**
```bash
git clone <your-repository-url>
cd financial-agent
```

2. Create the Environment File
Create a file named .env in the project's root directory. This file will securely store your API key. Add your OpenAI API key to it:
OPENAI_API_KEY="sk-..."

3. Place Data Files
Place the PDF documents (e.g., company 10-K filings) you want the agent to be able to search through into the /data directory.
4. Populate the Vector Database
This step reads the documents from the /data folder, processes them, and stores them in a local Chroma vector database. This only needs to be done once, or whenever you add new documents.
python ingest_data.py

This will create a chroma folder in your project directory containing the indexed data.
5. Build the Docker Image
This command reads the Dockerfile and builds a self-contained image with all the necessary code and dependencies.
docker build -t financial-agent .

Running the Application
Now that the setup is complete, you can run the application with a single command.
docker run -p 8000:8000 -v ./chroma:/app/chroma --env-file .env financial-agent

Understanding the Command:
-p 8000:8000: Maps port 8000 on your local machine to port 8000 inside the container.
-v ./chroma:/app/chroma: This is the volume mount. It links the chroma folder on your computer to the /app/chroma folder inside the container, giving the agent access to the database you created.
--env-file .env: Securely passes the environment variables (your API key) from your .env file to the container.
financial-agent: The name of the image to run.
You should see logs in your terminal indicating that the Uvicorn server has started and the RAG tool has successfully loaded the documents from your vector store.
Accessing the API
The easiest way to interact with the agent is through the auto-generated Swagger UI documentation.
1. Open your web browser and navigate to:
http://127.0.0.1:8000/docs
2. You will see the API documentation. Click on the POST /ask endpoint to expand it.
3. Click the "Try it out" button.
4. In the "Request body" text area, enter your question in JSON format. For example:
{
  "question": "What were the main business risks for Microsoft in its latest 10-K?"
}
5. Click the blue "Execute" button.
The agent will now process your request, and the response will appear on the page. You can watch the agent "think" in real-time in the terminal where the Docker container is running.

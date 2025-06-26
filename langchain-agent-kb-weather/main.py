
from langchain_aws.chat_models import ChatBedrock
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory

import boto3
import os
import requests 

from dotenv import load_dotenv

load_dotenv()
REGION = os.getenv("AWS_REGION", "us-east-1")
LLM_MODEL_ID = os.getenv("LLM_MODEL") 
EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL") 
# Step 1: Setup AWS credentials (use env or ~/.aws config normally)
boto3.setup_default_session(region_name=REGION)

# Step 2: Load your local text file (or PDF loader if needed)
loader = TextLoader("mydocs/sample.txt", encoding="utf-8")  # Replace with your file
documents = loader.load()

# Step 3: Split documents into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)

# Step 4: Setup Titan embedding model
embedding_model = BedrockEmbeddings(
    model_id=EMBEDDING_MODEL_ID,
    region_name=REGION
)

# Step 5: Create a Chroma vector store (locally)
chroma_db = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="./chroma_store"
)

# Step 6: Load the Claude model (Haiku is fast & cheap)
llm = ChatBedrock(
    model_id=LLM_MODEL_ID,
    region_name=REGION,
    model_kwargs={"temperature": 0.1}
)

retriever = chroma_db.as_retriever()

# ----- RAG TOOL -----
rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
kb_tool = Tool(
    name="KnowledgeBase",
    func=rag_chain.run,
    description="Use this to answer questions from the company documents or knowledge base."
)
# ----- WEATHER TOOL -----
def get_weather(city: str = "Ahmedabad"):
    try:
        # Free API from wttr.in (no API key required)
        response = requests.get(f"https://wttr.in/{city}?format=3")
        return response.text
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

weather_tool = Tool(
    name="CurrentWeather",
    func=get_weather,  # could parse query for city if needed
    description="Use this to fetch the current temperature in a city. Always call this for weather-related queries."
)

# ----- INIT AGENT -----
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
agent = initialize_agent(
    tools=[kb_tool, weather_tool],
    llm=llm,
    agent="chat-conversational-react-description",
    memory=memory,
    verbose=True
)

# ----- ASK SOMETHING -----

while True:
    user_input = input("\n💬 Ask me something about kb or weather (or type 'exit'): ")
    if user_input.lower() in ("exit", "quit"):
        break

    result = agent.invoke({
        "input": user_input
    })

    # Print response
    print("\n🤖 Agent:", result["output"])


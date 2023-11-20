import os
import sys
import time
import random

import openai
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

# Function to retry with exponential backoff
def retry_with_backoff(retries, base_delay=1, max_delay=60, backoff_factor=2):
    for i in range(retries):
        delay = min(base_delay * backoff_factor ** i, max_delay)
        delay = random.uniform(0, delay)
        yield delay

# Function to execute with retries
def execute_with_retries(func, max_retries=5, **kwargs):
    for delay in retry_with_backoff(retries=max_retries):
        try:
            return func(**kwargs)
        except Exception as e:
            print(f"Request failed: {e}, retrying in {delay} seconds...")
            time.sleep(delay)
    raise Exception("All retries failed")

# Load or create index
query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = TextLoader("data/data.txt")  # Use this line if you only need data.txt
    # loader = DirectoryLoader("data/")  # Uncomment this line if loading from directory
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory": "persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()

    # Use the execute_with_retries function to handle the call to chain
    try:
        result = execute_with_retries(
            func=chain,
            question={"question": query, "chat_history": chat_history}
        )
        print(result['answer'])
    except Exception as e:
        print(f"Failed to get an answer after several retries: {e}")
        break  # or handle it in some other way

    chat_history.append((query, result['answer']))
    query = None
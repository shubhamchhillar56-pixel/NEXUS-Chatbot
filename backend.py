
from __future__ import annotations
import os
import tempfile
import uuid
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage , RemoveMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg_pool import ConnectionPool

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

import psycopg

load_dotenv()

print("=" * 60)
print("GOOGLE_API_KEY =", repr(os.getenv("GOOGLE_API_KEY")))
print("DATABASE_URL =", repr(os.getenv("DATABASE_URL")))
print("ALL ENV KEYS =", sorted(os.environ.keys()))
print("=" * 60)

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY")

# Pass google_api_key explicitly into the class
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    GOOGLE_API_KEY=api_key,
    temperature=0.7
)
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    GOOGLE_API_KEY=api_key
)

# ==================== POSTGRESQL MEMORY SETUP ====================
# PostgreSQL Connection String (pointing to Docker port 5442)
DB_URI = os.getenv("DATABASE_URL")
if not DB_URI:
    raise ValueError("DATABASE_URL not found")

# 1. Enable autocommit on the pool connections
pool = ConnectionPool(
    conninfo=DB_URI, 
    max_size=20, 
    kwargs={"autocommit": True}  # Autocommit enabled across pool
)

# Initialize Checkpointer (Short-term) & Store (Long-term) with Postgres
checkpointer = PostgresSaver(pool)
in_memory_store = PostgresStore(pool)

# Setup database tables ONCE at startup
checkpointer.setup()
in_memory_store.setup()

# Setup custom thread_titles table
with pool.connection() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS thread_titles(
                thread_id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

# PDF retriever store (per thread)
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


# --- THREAD TITLE HELPER FUNCTIONS ---
def generate_chat_title(first_message: str) -> str:
    prompt = f"""
Generate a short title (maximum 5 words) for this conversation.

User message:
{first_message}

Return ONLY the title.
"""
    response = llm.invoke(prompt)
    return response.content.strip().replace('"', "")


def save_thread_title(thread_id: str, user_id: str, title: str):
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO thread_titles(thread_id, user_id, title)
                VALUES(%s, %s, %s)
                ON CONFLICT(thread_id)
                DO UPDATE SET title=EXCLUDED.title
                """,
                (str(thread_id), str(user_id), title)  # Cast to str
            )


def get_thread_title(thread_id) -> str:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT title
                FROM thread_titles
                WHERE thread_id=%s
                """,
                (str(thread_id),)  # Cast to str
            )
            row = cur.fetchone()
            if row:
                return row[0]
    return "New Chat"


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.
    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    summary: str  # Tracks the ongoing conversation summary

# --- Summarization Node Logic ---
def summarize_conversation_node(state: ChatState, config=None):
    """
    Summarizes older messages if the history gets too long.
    Retains the last 4 messages for immediate context and condenses the rest.
    """
    messages = state.get("messages", [])
    
    # Trigger summarization only when thread has more than 6 messages
    if len(messages) <= 6:
        return {}

    existing_summary = state.get("summary", "")
    
    # Keep the last 4 messages intact, summarize everything before them
    messages_to_summarize = messages[:-4]
    
    if existing_summary:
        summary_prompt = (
            f"This is a summary of the conversation so far: {existing_summary}\n\n"
            "Extend the summary by incorporating the following new messages:"
        )
    else:
        summary_prompt = "Create a succinct summary of the following conversation:"

    # Call LLM to create/update summary
    summary_messages = [SystemMessage(content=summary_prompt)] + messages_to_summarize
    response = llm.invoke(summary_messages)
    new_summary = response.content.strip()

    # Generate RemoveMessage operations to purge old messages from LangGraph checkpointer
    delete_messages = [RemoveMessage(id=m.id) for m in messages_to_summarize if hasattr(m, 'id') and m.id]

    return {
        "summary": new_summary,
        "messages": delete_messages  # Removes purged messages from state
    }
# --- LONG-TERM MEMORY TOOLS ---
@tool
def upsert_memory(memory_text: str, config: Optional[dict] = None) -> str:
    """
    Save or update an important fact, preference, or detail about the user 
    in long-term memory so it persists across different chat threads.
    """
    user_id = "default_user"
    if config and isinstance(config, dict):
        user_id = config.get("configurable", {}).get("user_id", "default_user")

    namespace = ("memories", user_id)
    memory_id = str(uuid.uuid4())
    
    in_memory_store.put(
        namespace=namespace,
        key=memory_id,
        value={"text": memory_text}
    )
    return f"Saved to long-term memory: '{memory_text}'"


@tool
def get_memories(config: Optional[dict] = None) -> list:
    """Fetch all saved long-term memories for the current user."""
    user_id = "default_user"
    if config and isinstance(config, dict):
        user_id = config.get("configurable", {}).get("user_id", "default_user")

    namespace = ("memories", user_id)
    memories = in_memory_store.search(namespace)
    return [m.value.get("text") for m in memories]


def get_all_user_memories(user_id: str = "default_user") -> list:
    """Helper function for Streamlit sidebar to fetch long-term memories."""
    namespace = ("memories", user_id)
    memories = in_memory_store.search(namespace)
    return [m.value.get("text") for m in memories if m.value.get("text")]


# --- STANDARD TOOLS ---
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    r = requests.get(url)
    return r.json()


tools = [search_tool, get_stock_price, calculator, rag_tool, upsert_memory, get_memories]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


def chat_node(state: ChatState, config=None):
    """LLM node that retrieves user memory, current summary, and responds."""
    thread_id = None
    user_id = "default_user"

    if config and isinstance(config, dict):
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        user_id = configurable.get("user_id", "default_user")

    # Fetch long-term memories
    namespace = ("memories", user_id)
    stored_memories = in_memory_store.search(namespace)
    memory_context = (
        "\n".join([f"- {m.value.get('text')}" for m in stored_memories])
        if stored_memories else "No prior memories recorded."
    )

    # Fetch running summary
    conversation_summary = state.get("summary", "No prior summary available.")

    system_message = SystemMessage(
        content=(
            "You are NEXUS, a helpful AI assistant.\n\n"
            f"=== USER LONG-TERM MEMORIES ===\n{memory_context}\n===============================\n\n"
            f"=== CONVERSATION SUMMARY ===\n{conversation_summary}\n============================\n\n"
            "Instructions:\n"
            "1. Use saved user memories and conversation summary to maintain full context.\n"
            "2. If the user shares an important personal fact or preference, use `upsert_memory` to store it.\n"
            f"3. For questions about uploaded PDFs, call `rag_tool` with thread_id `{thread_id}`.\n"
            "4. You also have access to web search, stock prices, and calculator tools."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}

# Graph Definition
graph = StateGraph(ChatState)

graph.add_node("summarize_node", summarize_conversation_node)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# Flow: START -> Summarize check -> Chat -> Tools or End
graph.add_edge(START, "summarize_node")
graph.add_edge("summarize_node", "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

# Compile graph
chatbot = graph.compile(
    checkpointer=checkpointer,
    store=in_memory_store,
    interrupt_before=["tools"]
)


def retrieve_all_threads() -> list[str]:
    """Retrieve unique thread IDs stored in PostgreSQL checkpoints."""
    allthread = set()
    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
        if thread_id:
            allthread.add(thread_id)
    return list(allthread)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})

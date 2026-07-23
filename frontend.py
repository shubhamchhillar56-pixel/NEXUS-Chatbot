import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend import (
    chatbot,
    in_memory_store,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
    get_thread_title,
    save_thread_title,
    generate_chat_title,
)


# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": str(thread_id)}})
    return state.values.get("messages", [])


def extract_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text = ""
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    text += block.get("text", "")
        return text

    return str(content)


# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "user_id" not in st.session_state:
    st.session_state["user_id"] = "default_user"  # Unique key for Long-Term Memory

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
user_key = st.session_state["user_id"]
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# Global configuration passed to LangGraph (contains both STM and LTM identifiers)
CONFIG = {
    "configurable": {
        "thread_id": thread_key,
        "user_id": user_key,
    },
    "metadata": {"thread_id": thread_key, "user_id": user_key},
    "run_name": "chat_turn",
}

# ============================ Sidebar ============================
st.sidebar.title("Shubham")
#st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")
st.sidebar.markdown(f"**User ID:** `{user_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

# Long-Term Memory Viewer
with st.sidebar.expander("🧠 Active Long-Term Memories", expanded=False):
    memories = in_memory_store.search(("memories", user_key))
    if memories:
        for m in memories:
            st.markdown(f"- {m.value.get('text')}")
    else:
        st.write("No memories saved yet.")

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"`{uploaded_pdf.name}` already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF…", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="✅ PDF indexed", state="complete", expanded=False)

st.sidebar.subheader("Past conversations")
if not threads:
    st.sidebar.write("No past conversations yet.")
else:
    for thread in threads:
        title = get_thread_title(thread)
        if st.sidebar.button(f"💬 {title}", key=f"thread-{thread}", use_container_width=True):
            selected_thread = thread

# --- PAST CONVERSATION ROUTER ROUTINE ---
if selected_thread and str(selected_thread) != thread_key:
    st.session_state["thread_id"] = selected_thread
    messages = load_conversation(selected_thread)

    temp_messages = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": extract_text(msg.content)})
    
    st.session_state["message_history"] = temp_messages
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()

# ============================ Main Layout ========================
st.title("NEXUS Chatbot")

# Render historical messages
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- HUMAN-IN-THE-LOOP INTERRUPT CHECK ---
current_state = chatbot.get_state(config=CONFIG)

if current_state.next: 
    st.warning("⚠️ The chatbot is waiting for approval to run an external tool.")
    
    last_message = current_state.values.get("messages", [])[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        st.json(last_message.tool_calls)

    if st.button("👍 Approve & Proceed", use_container_width=True):
        with st.chat_message("assistant"):
            status_holder = {"box": None}
            
            def ai_resume_stream():
                for message_chunk, _ in chatbot.stream(
                    None, 
                    config=CONFIG,
                    stream_mode="messages",
                ):
                    if isinstance(message_chunk, ToolMessage):
                        tool_name = getattr(message_chunk, "name", "tool")
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(f"🔧 Using `{tool_name}` …", expanded=True)
                    if isinstance(message_chunk, AIMessage):
                        yield extract_text(message_chunk.content)

            ai_message = st.write_stream(ai_resume_stream())
            
            if status_holder["box"] is not None:
                status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)
        
        st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
        st.rerun()

# --- STANDARD USER CHAT INPUT ---
user_input = st.chat_input("Ask anything")

if user_input:
    # Title Generation for New Chats
    if get_thread_title(thread_key) == "New Chat":
        title = generate_chat_title(user_input)
        save_thread_title(thread_key, user_key, title)

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(f"🔧 Using `{tool_name}` …", expanded=True)
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                if isinstance(message_chunk, AIMessage):
                    yield extract_text(message_chunk.content)

        ai_message = st.write_stream(ai_only_stream())

        if status_holder["box"] is not None:
            status_holder["box"].update(label="✅ Tool finished", state="complete", expanded=False)

    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})
    
    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )
    st.rerun()

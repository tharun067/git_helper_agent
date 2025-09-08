import streamlit as st
import time
from typing import Generator, Dict, Any
import random
import os
from datetime import datetime

# Import your existing modules (uncomment when ready to use)
from vector_store import VectorStore, GitVectorStore, CHROMADB_PATH, COLLECTION_NAME, CHUNK_SIZE, EMBEDDING_MODEL, \
    CHUNK_OVERLAP
from llm_model import agent_build

# Page configuration
st.set_page_config(
    page_title="GitMind AI Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }

    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    .user-message {
        background: linear-gradient(90deg, #a8edea 0%, #fed6e3 100%);
        margin-left: 2rem;
    }

    .bot-message {
        background: linear-gradient(90deg, #d299c2 0%, #fef9d7 100%);
        margin-right: 2rem;
    }

    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }

    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .status-connected { background-color: #28a745; }
    .status-disconnected { background-color: #dc3545; }
    .status-processing { background-color: #ffc107; }

    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'repository_status' not in st.session_state:
    st.session_state.repository_status = "Not Connected"
if 'vector_store_ready' not in st.session_state:
    st.session_state.vector_store_ready = False
if 'processing' not in st.session_state:
    st.session_state.processing = False


def get_status_color(status):
    status_colors = {
        "Not Connected": "status-disconnected",
        "Connected": "status-connected",
        "Processing": "status-processing"
    }
    return status_colors.get(status, "status-disconnected")


def add_to_chat(role, message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append({
        "role": role,
        "message": message,
        "timestamp": timestamp
    })


def validate_github_url(url):
    """Validate GitHub URL format"""
    if not url:
        return False, "URL cannot be empty"
    if not url.startswith(("https://github.com/", "git@github.com:")):
        return False, "Please enter a valid GitHub URL"
    return True, "Valid URL"


# Header
st.markdown("""
<div class="main-header">
    <h1>🚀 GitMind AI Assistant</h1>
    <p>Intelligent code analysis and Q&A for your GitHub repositories</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📋 Repository Settings")

    # Repository Status
    st.markdown(f"""
    <div class="sidebar-section">
        <h4>Repository Status</h4>
        <span class="status-indicator {get_status_color(st.session_state.repository_status)}"></span>
        <span>{st.session_state.repository_status}</span>
    </div>
    """, unsafe_allow_html=True)

    # Repository Configuration
    st.markdown("### 🔗 Repository Configuration")

    remote_url = st.text_input(
        "GitHub Remote URL",
        placeholder="https://github.com/username/repository.git",
        help="Enter the HTTPS or SSH URL of your GitHub repository"
    )

    local_path = st.text_input(
        "Local Clone Path",
        placeholder="./repositories/my-repo",
        help="Specify where to clone the repository locally"
    )

    # Advanced Settings (collapsible)
    with st.expander("⚙️ Advanced Settings"):
        chunk_size = st.slider("Chunk Size", 100, 2000, CHUNK_SIZE)
        chunk_overlap = st.slider("Chunk Overlap", 0, 500, CHUNK_OVERLAP)
        st.info("These settings control how your code is processed for analysis")

    # Action buttons
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Connect Repository", key="connect_repo"):
            if remote_url and local_path:
                is_valid, message = validate_github_url(remote_url)
                if is_valid:
                    st.session_state.processing = True
                    st.session_state.repository_status = "Processing"

                    with st.spinner("Connecting to repository..."):
                        try:
                            # Simulate processing time
                            progress_bar = st.progress(0)
                            for i in range(101):
                                progress_bar.progress(i)
                                time.sleep(0.01)

                            # Initialize the pipeline
                            pipeline = GitVectorStore(
                                repo_url=remote_url,
                                clone_path=local_path,
                                persist_dir=CHROMADB_PATH,
                                collection_name=COLLECTION_NAME,
                                model_name=EMBEDDING_MODEL,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap
                            )
                            pipeline.run_pipeline()

                            st.session_state.repository_status = "Connected"
                            st.session_state.vector_store_ready = True
                            st.session_state.processing = False
                            st.success("✅ Repository connected successfully!")
                            add_to_chat("system", f"Connected to repository: {remote_url}")

                        except Exception as e:
                            st.error(f"❌ Error connecting to repository: {str(e)}")
                            st.session_state.repository_status = "Not Connected"
                            st.session_state.processing = False
                else:
                    st.error(f"❌ {message}")
            else:
                st.warning("⚠️ Please fill in both URL and local path")

    with col2:
        if st.button("🗑️ Clear History", key="clear_history"):
            try:
                VectorStore.clear_collection()
                st.session_state.chat_history = []
                st.session_state.repository_status = "Not Connected"
                st.session_state.vector_store_ready = False
                st.success("✅ History cleared successfully!")
            except Exception as e:
                st.error(f"❌ Error clearing history: {str(e)}")

    # Statistics
    if st.session_state.chat_history:
        st.markdown("### 📊 Session Stats")
        total_questions = len([msg for msg in st.session_state.chat_history if msg["role"] == "user"])
        st.markdown(f"""
        <div class="metric-card">
            <h3>{total_questions}</h3>
            <p>Questions Asked</p>
        </div>
        """, unsafe_allow_html=True)

# Main chat interface
st.markdown("## 💬 Chat with Your Repository")

# Chat container
chat_container = st.container()

# Display chat history
with chat_container:
    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🧑‍💻 You ({chat['timestamp']})</strong><br>
                    {chat['message']}
                </div>
                """, unsafe_allow_html=True)
            elif chat["role"] == "assistant":
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 GitMind ({chat['timestamp']})</strong><br>
                    {chat['message']}
                </div>
                """, unsafe_allow_html=True)
            elif chat["role"] == "system":
                st.info(f"ℹ️ {chat['message']}")
    else:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #666;">
            <h3>👋 Welcome to GitMind!</h3>
            <p>Connect your repository and start asking questions about your code.</p>
        </div>
        """, unsafe_allow_html=True)

# Question input
st.markdown("### ❓ Ask a Question")
question_col1, question_col2 = st.columns([4, 1])

with question_col1:
    user_question = st.text_input(
        "Enter your question:",
        placeholder="e.g., 'How does the authentication system work?' or 'Show me the main components'",
        key="user_question"
    )

with question_col2:
    ask_button = st.button("🚀 Ask", key="ask_question")

# Process question
if (ask_button or user_question) and user_question:
    if not st.session_state.vector_store_ready:
        st.warning("⚠️ Please connect to a repository first!")
    else:
        add_to_chat("user", user_question)

        with st.spinner("🤔 Thinking..."):
            try:
                # Initialize the agent
                graph = agent_build()

                # Get response from the agent
                response = graph.invoke({"messages": [("user", user_question)]})

                # Extract the answer
                if hasattr(response["messages"][-1], "content"):
                    answer = response["messages"][-1].content
                else:
                    answer = response["messages"][-1]["content"]

                add_to_chat("assistant", answer)
                st.rerun()

            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                add_to_chat("assistant", error_message)
                st.error(f"❌ {error_message}")

# Quick action buttons
if st.session_state.vector_store_ready:
    st.markdown("### 🔥 Quick Questions")
    quick_questions = [
        "📁 What is the overall structure of this repository?",
        "🔧 What are the main functions and their purposes?",
        "📝 Can you explain the README file?",
        "🐛 Are there any potential issues in the code?",
        "🚀 How can I get started with this project?"
    ]

    cols = st.columns(len(quick_questions))
    for i, question in enumerate(quick_questions):
        with cols[i]:
            if st.button(question, key=f"quick_{i}"):
                add_to_chat("user", question.split(" ", 1)[1])  # Remove emoji
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Built with ❤️ using Streamlit | GitMind AI Assistant v2.0</p>
</div>
""", unsafe_allow_html=True)
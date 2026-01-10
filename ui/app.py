"""
Streamlit UI for fraud chatbot.
"""

import streamlit as st
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from llm.qa_chain import QAChain


# Page configuration
st.set_page_config(
    page_title="Fraud Detection Chatbot",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .question-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .answer-box {
        background-color: #f1f8e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_qa_chain():
    """Load QA chain (cached)."""
    return QAChain(llm_provider="openai", model_name="gpt-3.5-turbo")


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Fraud Detection Q&A Chatbot</h1>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("⚙️ Configuration")
    
    num_sources = st.sidebar.slider("Number of sources to retrieve", 1, 10, 3)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "🤖 **Model:** GPT-3.5 Turbo (OpenAI)\n\n"
        "This chatbot uses RAG (Retrieval-Augmented Generation) to answer "
        "questions about fraud transactions using your dataset and documents."
    )
    
    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'qa_chain' not in st.session_state:
        try:
            with st.spinner("Loading QA system..."):
                st.session_state.qa_chain = load_qa_chain()
            st.success("✅ QA system loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading QA system: {str(e)}")
            st.stop()
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, (chunk, score) in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}** (Relevance: {score:.3f})")
                        st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                        st.markdown("---")
    
    # Chat input
    if question := st.chat_input("Ask a question about fraud transactions..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        
        with st.chat_message("user"):
            st.markdown(question)
        
        # Get answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.qa_chain.ask(question, k=num_sources)
                    answer = result["answer"]
                    sources = result["sources"]
                    
                    st.markdown(answer)
                    
                    # Show sources
                    with st.expander("📚 View Sources"):
                        for i, (chunk, score) in enumerate(sources, 1):
                            st.markdown(f"**Source {i}** (Relevance: {score:.3f})")
                            st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                            st.markdown("---")
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Clear chat button
    if st.sidebar.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


if __name__ == "__main__":
    main()

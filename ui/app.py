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
    
    # Display dynamic settings from QA chain
    if 'qa_chain' in st.session_state:
        threshold = st.session_state.qa_chain.relevance_threshold
        max_chunks = st.session_state.qa_chain.max_chunks
    else:
        threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.7"))
        max_chunks = int(os.getenv("MAX_CHUNKS", "10"))
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"🤖 **Model:** GPT-3.5 Turbo (OpenAI)\n\n"
        f"📚 **Retrieval:** Dynamic\n"
        f"- Max chunks: {max_chunks}\n"
        f"- Relevance threshold: {threshold}\n"
        f"- Max context: 5 chunks\n\n"
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
            
            if "sources" in message and message["sources"]:
                num_sources = len(message["sources"])
                with st.expander(f"📚 View Sources ({num_sources} chunks)"):
                    for i, source_item in enumerate(message["sources"], 1):
                        if len(source_item) == 3:
                            chunk, score, metadata = source_item
                            meta_str = ""
                            if metadata:
                                meta_str = f" - {metadata.get('source', 'unknown')}, Page {metadata.get('page', 'N/A')}"
                            st.markdown(f"**Source {i}**{meta_str} (Relevance: {score:.3f})")
                        else:
                            chunk, score = source_item
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
                    result = st.session_state.qa_chain.ask(question)
                    answer = result["answer"]
                    sources = result["sources"]
                    num_chunks = result.get("num_chunks_used", 0)
                    threshold = result.get("relevance_threshold", 0.7)
                    
                    st.markdown(answer)
                    
                    # Show sources
                    with st.expander(f"📚 View Sources ({num_chunks} chunks used, threshold: {threshold})"):
                        if sources:
                            for i, source_item in enumerate(sources, 1):
                                if len(source_item) == 3:
                                    chunk, score, metadata = source_item
                                    meta_str = ""
                                    if metadata:
                                        meta_str = f" - {metadata.get('source', 'unknown')}, Page {metadata.get('page', 'N/A')}"
                                    st.markdown(f"**Source {i}**{meta_str} (Relevance: {score:.3f})")
                                else:
                                    chunk, score = source_item
                                    st.markdown(f"**Source {i}** (Relevance: {score:.3f})")
                                st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
                                st.markdown("---")
                        else:
                            st.info("No sources met the relevance threshold.")
                    
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

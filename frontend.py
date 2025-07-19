import streamlit as st
from query_engine import QueryEngine
from document_processor import DocumentProcessor

# Set page config
st.set_page_config(
    page_title="Company Q&A Assistant",
    page_icon="🤖",
    layout="wide"
)

def main():
    st.title("🤖 Company Documents Q&A Assistant")
    st.write("Ask me anything about company policies and procedures!")
    
    # Initialize components
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = QueryEngine()
    
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📚 Document Management")
        
        if st.button("Load Sample Documents"):
            with st.spinner("Loading sample documents..."):
                st.session_state.doc_processor.add_sample_documents()
                st.success("Sample documents loaded!")
        
        st.write("---")
        st.header("🤖 Model Settings")
        use_advanced = st.checkbox("Use Advanced AI Generation", help="Uses HuggingFace model for responses (experimental)")
        
        if use_advanced:
            st.info("⚡ Advanced mode uses local HuggingFace models")
        else:
            st.info("🚀 Simple mode uses template-based responses (faster)")
        
        st.write("---")
        st.write("**Available sample questions:**")
        st.write("- What's our vacation policy?")
        st.write("- How do I submit an expense report?")
        st.write("- Who should I contact for IT issues?")
        st.write("- How many vacation days do I get?")
    
    # Main chat interface
    st.header("💬 Ask a Question")
    
    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Question input
    question = st.text_input(
        "Type your question here:",
        placeholder="e.g., What's our vacation policy?"
    )
    
    if st.button("Ask") and question:
        with st.spinner("Searching documents and generating answer..."):
            # Get answer with selected mode
            answer = st.session_state.query_engine.ask_question(question, use_advanced)
            
            # Add to chat history
            st.session_state.chat_history.append({
                'question': question,
                'answer': answer
            })
    
    # Display chat history
    if st.session_state.chat_history:
        st.header("💭 Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['question']}", expanded=(i==0)):
                st.write("**Answer:**")
                st.write(chat['answer'])

if __name__ == "__main__":
    main()
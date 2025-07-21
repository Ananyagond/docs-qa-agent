import streamlit as st
from query_engine import QueryEngine
from document_processor import DocumentProcessor
import os
import tempfile
from pathlib import Path
import docx
import PyPDF2
import pandas as pd
from io import StringIO
import time
from datetime import datetime

# Set page config with modern styling
st.set_page_config(
    page_title=" NovaDocs Q&A Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
def load_css():
    st.markdown("""
    <style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        margin: 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Quick action buttons */
    .quick-action {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .quick-action:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Chat message styling */
    .chat-message {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    .question-text {
        color: #667eea;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .answer-text {
        color: #475569;
        line-height: 1.6;
        margin-bottom: 0.5rem;
    }
    
    .source-info {
        color: #64748b;
        font-size: 0.9rem;
        font-style: italic;
        background: rgba(102, 126, 234, 0.1);
        padding: 0.5rem;
        border-radius: 8px;
        margin-top: 0.5rem;
    }
    
    /* File upload area */
    .upload-area {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        margin: 1rem 0;
    }
    
    /* Success/Error messages */
    .success-message {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .error-message {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* Statistics cards */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Progress bar */
    .progress-container {
        background: #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-bar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        height: 8px;
        transition: width 0.3s ease;
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .custom-footer {
        padding: 16px 20px;
        text-align: center;
        font-size: 0.9rem;
        background-color: #1f2937;
        color: #9ca3af;
        border-top: 1px solid #374151;
        border-radius: 12px;
        width: fit-content;
        max-width: 90%;
        transition: all 0.3s ease;
    }

    .custom-footer strong {
        color: #f3f4f6;
    }

    .custom-footer .emoji {
        margin-right: 4px;
    }

    .custom-footer:hover {
        background-color: #111827;
        color: #f9fafb;
    }

                
    /* Hide streamlit style */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    </style>
    """, unsafe_allow_html=True)

# Enhanced document management
def show_document_manager():
    st.markdown("### 📋 Document Database")
    
    # Show uploaded documents with better styling
    if os.path.exists("uploads"):
        uploaded_files = os.listdir("uploads")
        if uploaded_files:
            st.markdown("📄 Uploaded Documents:")
            for file in uploaded_files:
                col1, col2, col3 = st.columns([4, 1, 1])
                with col1:
                    st.write(f"📄 {file}")
                with col2:
                    file_size = os.path.getsize(f"uploads/{file}")
                    st.write(f"{file_size // 1024} KB")
                with col3:
                    if st.button("🗑", key=f"delete_{file}", help="Delete file"):
                        os.remove(f"uploads/{file}")
                        st.success(f"✅ Deleted {file}")
                        st.rerun()
        else:
            st.info("📁 No documents uploaded yet")

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"❌ Error reading DOCX file: {str(e)}")
        return None

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"❌ Error reading PDF file: {str(e)}")
        return None

def extract_text_from_csv(file):
    """Extract text from CSV file"""
    try:
        df = pd.read_csv(file)
        text = f"CSV Data from {file.name}:\n\n"
        text += f"Columns: {', '.join(df.columns)}\n\n"
        text += f"Total rows: {len(df)}\n\n"
        text += "Sample data:\n"
        text += df.head(10).to_string(index=False)
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            text += "\n\nSummary Statistics:\n"
            text += df[numeric_cols].describe().to_string()
        
        return text
    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        return None

def process_uploaded_file(uploaded_file, doc_processor):
    """Process an uploaded file and add to vector store"""
    try:
        os.makedirs("uploads", exist_ok=True)
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        # Extract text based on file type
        text_content = None
        
        if file_extension == '.txt':
            text_content = str(uploaded_file.read(), "utf-8")
        elif file_extension == '.docx':
            text_content = extract_text_from_docx(uploaded_file)
        elif file_extension == '.pdf':
            text_content = extract_text_from_pdf(uploaded_file)
        elif file_extension == '.csv':
            text_content = extract_text_from_csv(uploaded_file)
        else:
            st.error(f"❌ Unsupported file type: {file_extension}")
            return False
        
        if not text_content or len(text_content.strip()) == 0:
            st.error("❌ Could not extract text from the file or file is empty")
            return False
        
        # Save the extracted content
        file_path = f"uploads/{uploaded_file.name}_processed.txt"
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        # Process the file
        document_id = Path(uploaded_file.name).stem
        success = doc_processor.process_text_file(file_path, document_id)
        
        if success:
            st.success(f"✅ Successfully processed '{uploaded_file.name}'!")
            st.info(f"📄 Extracted {len(text_content):,} characters of text")
            return True
        else:
            st.error("❌ Failed to process the document")
            return False
            
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        return False

def display_chat_message(question, answer, sources=None, timestamp=None):
    """Display a chat message with modern styling"""

    timestamp_str = ""
    if timestamp:
        timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")

    st.markdown(f"""
    <div class="chat-message" style="background-color: #f9f9f9; padding: 1rem; border-radius: 12px; margin-bottom: 1rem;">
        <div style="color: #0f172a;"><strong>❓ Q:</strong> {question}</div>
        <div style="margin-top: 0.5rem; color: #1e293b;"><strong>💡 Answer:</strong><br>{answer}</div>
        {"<div style='color: gray; font-size: 0.85rem; margin-top: 0.5rem;'>🕒 " + timestamp_str + "</div>" if timestamp_str else ""}
        {f'<div style="color: #475569; font-size: 0.9rem; margin-top: 0.5rem;">📄 Source: {sources}</div>' if sources else ''}
    </div>
    """, unsafe_allow_html=True)


def show_footer():
    st.markdown("""
    <div style="display: flex; justify-content: center;">
        <div class="custom-footer">
            <span class="emoji">🤖</span><strong>AI Agent Hackathon 2025</strong> organized by <strong>Product Space</strong><br>
            🚀 Built with ❤️ by Team Innovators
            <span class="emoji">🚀</span> This project was built by our team: <strong>Taiyaba Khan</strong> & <strong>Ananya Gond</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    load_css()

    st.markdown("""
    <div class="main-header">
        <h1>🤖 NovaDocs Q&A Assistant</h1>
        <p>Ask me anything about company policies and procedures!</p>
    </div>
    """, unsafe_allow_html=True)

    if 'query_engine' not in st.session_state:
        with st.spinner("🔄 Initializing AI models..."):
            st.session_state.query_engine = QueryEngine()

    if 'doc_processor' not in st.session_state:
        with st.spinner("🔄 Setting up document processor..."):
            st.session_state.doc_processor = DocumentProcessor()

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if "chat_input_textarea" not in st.session_state:
        st.session_state.chat_input_textarea = ""

    if "set_input_value" not in st.session_state:
        st.session_state.set_input_value = None

    if st.session_state.set_input_value is not None:
        st.session_state.chat_input_textarea = st.session_state.set_input_value
        st.session_state.set_input_value = None

    # Sidebar
    with st.sidebar:
        st.markdown("## 📚 Document Management")

        st.markdown("### 🎯 Quick Start")
        if st.button("📥 Load Sample Documents", use_container_width=True):
            with st.spinner("Loading sample documents..."):
                st.session_state.doc_processor.add_sample_documents()
                st.success("✅ Sample documents loaded!")

        st.markdown("---")

        st.markdown("### 📤 Upload Your Documents")
        uploaded_files = st.file_uploader(
            "Choose files to upload", type=['txt', 'pdf', 'docx', 'csv'],
            accept_multiple_files=True,
            help="📋 Supported formats: TXT, PDF, DOCX, CSV"
        )

        if uploaded_files:
            st.markdown(f"📁 Selected files ({len(uploaded_files)}):")
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size / 1024:.1f} KB)")

            if st.button("🚀 Process All Files", use_container_width=True):
                progress_bar = st.progress(0)
                success_count = 0
                for i, file in enumerate(uploaded_files):
                    st.write(f"⏳ Processing: {file.name}...")
                    if process_uploaded_file(file, st.session_state.doc_processor):
                        success_count += 1
                    progress_bar.progress((i + 1) / len(uploaded_files))
                if success_count == len(uploaded_files):
                    st.balloons()
                    st.success(f"🎉 Successfully processed all {success_count} files!")
                else:
                    st.warning(f"⚠ Processed {success_count}/{len(uploaded_files)} files")

        st.markdown("---")

        st.markdown("### 📊 Database Statistics")
        col1, col2 = st.columns(2)
        if st.button("🔄 Refresh Stats", use_container_width=True):
            try:
                stats = st.session_state.query_engine.get_vector_store_stats()
                with col1:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-number">{stats.get('total_documents', 0)}</div>
                        <div class="stat-label">Documents</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="stat-card">
                        <div class="stat-number">{stats.get('dimension', 768)}</div>
                        <div class="stat-label">Vector Dim</div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error fetching stats: {str(e)}")

        if st.button("🗑 Clear All Documents", use_container_width=True):
            if st.checkbox("⚠ I confirm I want to clear all documents"):
                st.session_state.doc_processor.clear_all_documents()
                st.success("🧹 All documents cleared!")

        st.markdown("---")

        st.markdown("### 🤖 Model Settings")
        use_advanced = st.checkbox("🧠 Use Advanced AI Generation", help="Uses HuggingFace model for responses (experimental)")
        st.info("⚡ Advanced mode: Local HuggingFace models" if use_advanced else "🚀 Simple mode: Template-based responses (faster)")

        st.markdown("---")

        st.markdown("### 💡 Sample Questions")
        sample_questions = [
            "What's our vacation policy?",
            "How do I submit an expense report?", 
            "Who should I contact for IT issues?",
            "How many vacation days do I get?"
        ]
        with st.expander("💡 Sample Questions"):
            for q in sample_questions:
                if st.button(f"💬 {q}", key=f"sample_{q}", use_container_width=True):
                    st.session_state.set_input_value = q
                    st.rerun()

    # Main content
    tab1, tab2 = st.tabs(["🗨️ Chat Assistant", "📚 Instructions"])

    with tab1:
        st.markdown("### 🎯 Quick Questions")
        quick_questions = [
            {"label": "📋 Vacation Policy", "value": "What's our vacation policy?"},
            {"label": "💰 Expense Reports", "value": "How do I submit an expense report?"},
            {"label": "🔧 IT Support", "value": "Who should I contact for IT issues?"},
            {"label": "📅 Time Off Request", "value": "How do I request time off?"},
            {"label": "🧾 Reimbursement Rules", "value": "What are the expense reimbursement rules?"},
            {"label": "🙋‍♂️ HR Contact", "value": "Who is my HR contact?"}
        ]

        for i in range(0, len(quick_questions), 3):
            cols = st.columns(3)
            for col, q in zip(cols, quick_questions[i:i+3]):
                with col:
                    if st.button(q["label"], use_container_width=True, key=f"quick_{q['label']}"):
                        st.session_state.set_input_value = q["value"]
                        st.rerun()

        question = st.text_area(
            "Type your question here:",
            key="chat_input_textarea",
            placeholder="e.g., What's our vacation policy?",
            height=100
        )

        col_ask, col_clear = st.columns([3, 1])
        with col_ask:
            ask_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)

        with col_clear:
            if st.button("🧹 Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.success("✅ Chat cleared!")
                st.rerun()

        if ask_button and question.strip():
            with st.spinner("🤔 Searching documents and generating answer..."):
                try:
                    answer = st.session_state.query_engine.ask_question(question, use_advanced)
                    st.session_state.chat_history.append({
                        'question': question,
                        'answer': answer,
                        'timestamp': time.time()
                    })
                    st.session_state.set_input_value = ""  # Clear input safely
                    st.success("✅ Answer generated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")

        if st.session_state.chat_history:
            st.markdown("## 💭 Chat History")
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"Q: {chat['question'][:50]}...", expanded=(i == 0)):
                    display_chat_message(
                        question=chat['question'],
                        answer=chat['answer'],
                        sources="Processed documents",
                        timestamp=chat.get('timestamp')
                    )
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #64748b;">
                <h3>🤖 Ready to help!</h3>
                <p>Ask me anything about your company documents</p>
            </div>
            """, unsafe_allow_html=True)


    with tab2:
        with st.expander("📋 How to Use This Assistant", expanded=False):
            instructions_content = """
            ### 🚀 How to use this assistant:
            
            *1. 📤 Upload Documents*
            - Use the sidebar to upload your company documents
            - Supported: PDF, DOCX, TXT, CSV files
            
            *2. ⚡ Process Files* 
            - Click "Process All Files" to add them to the knowledge base
            - Wait for processing to complete
            
            *3. ❓ Ask Questions*
            - Type your questions in natural language
            - Use the quick question buttons for common queries
            
            *4. 🎯 Get Answers*
            - The AI searches through your documents
            - Provides relevant, context-aware answers
            
            ### 📁 Supported File Types:
            - 📄 *TXT*: Plain text files
            - 📑 *PDF*: PDF documents  
            - 📝 *DOCX*: Word documents
            - 📊 *CSV*: Spreadsheet data
            
            ### 💡 Tips for Better Results:
            - ✅ Be specific in your questions
            - ✅ Use keywords from your documents  
            - ✅ Try different phrasings if needed
            - ✅ Upload relevant, well-structured documents
            """
            
            st.markdown(instructions_content)
            
            st.markdown("### 🎯 Example Questions")
            example_questions = [
                "What is the company's remote work policy?",
                "How do I request time off?", 
                "What are the expense reimbursement rules?",
                "Who is my HR contact?",
                "What benefits does the company offer?",
                "How do I access the VPN?"
            ]
            
            for q in example_questions:
                st.markdown(f"• {q}")
    show_footer()

if __name__ == "__main__":
    main()
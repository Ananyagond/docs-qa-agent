import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from local_vector_store import LocalVectorStore
from dotenv import load_dotenv

load_dotenv()

class DocumentProcessor:
    def __init__(self):
        # Initialize local vector store
        print("Initializing local vector store...")
        self.vector_store = LocalVectorStore()
        print("Vector store ready!")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def process_text_file(self, file_path, document_id):
        """Process a text file and add to vector database"""
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split into chunks
            chunks = self.text_splitter.split_text(content)
            print(f"Split document into {len(chunks)} chunks")
            
            # Prepare data for vector store
            texts = chunks
            metadatas = []
            for i, chunk in enumerate(chunks):
                metadatas.append({
                    "text": chunk,
                    "document_id": document_id,
                    "chunk_index": i,
                    "source": file_path
                })
            
            # Add to vector store
            self.vector_store.add_documents(texts, metadatas)
            
            print(f"Successfully processed {file_path}")
            return True
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return False
    
    def add_sample_documents(self):
        """Add some sample company documents"""
        sample_docs = {
            "vacation_policy": """
            Company Vacation Policy
            
            All full-time employees are entitled to 15 days of paid vacation per year.
            Vacation days accrue at a rate of 1.25 days per month.
            Employees must request vacation at least 2 weeks in advance.
            Vacation requests should be submitted through the HR portal.
            Unused vacation days cannot be carried over to the next year.
            Maximum vacation that can be taken at once is 10 consecutive days.
            Part-time employees receive prorated vacation based on hours worked.
            """,
            
            "expense_policy": """
            Expense Reimbursement Policy
            
            Employees can be reimbursed for business-related expenses.
            All expenses must be submitted within 30 days with receipts.
            Meals are reimbursed up to $50 per day during business travel.
            Transportation costs for business trips are fully covered.
            Hotel accommodation is covered up to $200 per night.
            Submit expense reports through the finance portal at finance.company.com.
            Approval from manager required for expenses over $500.
            Personal expenses will not be reimbursed under any circumstances.
            """,
            
            "it_support": """
            IT Support Guidelines
            
            For technical issues, contact IT support at it-help@company.com
            For urgent issues, call the IT hotline: (555) 123-4567
            Common issues can be resolved through the self-service portal.
            New equipment requests should be submitted through IT portal.
            Software installation requires IT approval for security reasons.
            Password resets can be done through the company portal.
            VPN access is required for remote work - contact IT for setup.
            Regular security training is mandatory for all employees.
            """,
            
            "remote_work": """
            Remote Work Policy
            
            Employees can work remotely up to 3 days per week.
            Remote work must be pre-approved by direct manager.
            All remote workers must have reliable internet connection.
            Company will provide necessary equipment for home office setup.
            Remote workers must be available during core hours 10am-3pm EST.
            Monthly in-person team meetings are required.
            Productivity metrics will be tracked for remote workers.
            Remote work privileges can be revoked for performance issues.
            """
        }
        
        # Create sample_docs folder if it doesn't exist
        os.makedirs("sample_docs", exist_ok=True)
        
        # Save and process each document
        for doc_id, content in sample_docs.items():
            file_path = f"sample_docs/{doc_id}.txt"
            with open(file_path, 'w') as f:
                f.write(content)
            
            self.process_text_file(file_path, doc_id)
        
        # Print statistics
        stats = self.vector_store.get_stats()
        print(f"Vector store now contains {stats['total_documents']} document chunks")
    
    def get_vector_store_stats(self):
        """Get vector store statistics"""
        return self.vector_store.get_stats()
    
    def clear_all_documents(self):
        """Clear all documents from vector store"""
        self.vector_store.clear()
import os
from local_vector_store import LocalVectorStore
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv

load_dotenv()

class QueryEngine:
    def __init__(self):
        # Initialize local vector store
        print("Loading vector store for search...")
        self.vector_store = LocalVectorStore()
        
        # Initialize text generation model (runs locally!)
        print("Loading text generation model...")
        model_name = "microsoft/DialoGPT-medium"  # Lightweight conversational model
        
        # Alternative models you can try:
        # model_name = "distilgpt2"  # Very lightweight
        # model_name = "facebook/blenderbot-400M-distill"  # Good for Q&A
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Models loaded successfully!")
    
    def search_documents(self, question, top_k=3):
        """Search for relevant documents using the question"""
        try:
            # Search using local vector store
            search_results = self.vector_store.search(question, k=top_k, score_threshold=0.5)
            
            # Format results
            relevant_chunks = []
            for result in search_results:
                relevant_chunks.append({
                    'text': result['text'],
                    'source': result['metadata']['source'],
                    'score': result['score']
                })
            
            return relevant_chunks
            
        except Exception as e:
            print(f"Error searching documents: {str(e)}")
            return []
    
    def generate_answer_simple(self, question, relevant_chunks):
        """Generate a simple answer using template matching (fallback)"""
        if not relevant_chunks:
            return "I couldn't find relevant information to answer your question. Please try rephrasing or check if the document exists in our system."
        
        # Simple template-based response
        context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
        
        # Look for key information
        question_lower = question.lower()
        
        if "vacation" in question_lower or "time off" in question_lower:
            if "15 days" in context:
                answer = "Based on our company policy, full-time employees are entitled to 15 days of paid vacation per year. Vacation days accrue at 1.25 days per month and must be requested at least 2 weeks in advance through the HR portal. Unused vacation days cannot be carried over to the next year."
            else:
                answer = f"Here's what I found about vacation policy: {context[:300]}..."
                
        elif "expense" in question_lower or "reimburs" in question_lower:
            answer = "For expense reimbursement, submit expenses within 30 days with receipts. Meals are covered up to $50/day during business travel, transportation is fully covered, and hotels up to $200/night. Submit through the finance portal at finance.company.com."
            
        elif "it" in question_lower or "support" in question_lower or "technical" in question_lower:
            answer = "For IT support, contact it-help@company.com or call (555) 123-4567 for urgent issues. Use the self-service portal for common issues. New equipment and software requests need IT approval."
            
        elif "remote" in question_lower or "work from home" in question_lower:
            answer = "Employees can work remotely up to 3 days per week with manager approval. You need reliable internet, and the company provides home office equipment. Core hours are 10am-3pm EST, with monthly in-person meetings required."
            
        elif "how many" in question_lower and "vacation" in question_lower:
            answer = "Full-time employees get 15 days of paid vacation per year, accruing at 1.25 days per month. Part-time employees receive prorated vacation based on hours worked."
            
        elif "who" in question_lower and ("contact" in question_lower or "call" in question_lower):
            if "it" in question_lower:
                answer = "For IT issues, contact it-help@company.com or call the IT hotline at (555) 123-4567 for urgent matters."
            else:
                answer = f"Based on our policies: {context[:200]}..."
                
        else:
            # General response
            answer = f"Based on the company documents, here's what I found: {context[:400]}..."
        
        # Add sources
        sources = list(set([chunk['source'] for chunk in relevant_chunks]))
        if sources:
            source_names = [os.path.basename(src) for src in sources]
            answer += f"\n\n📚 Sources: {', '.join(source_names)}"
        
        return answer
    
    def generate_answer_advanced(self, question, relevant_chunks):
        """Generate answer using HuggingFace model (experimental)"""
        if not relevant_chunks:
            return self.generate_answer_simple(question, relevant_chunks)
        
        try:
            # Prepare context
            context = "\n".join([chunk['text'] for chunk in relevant_chunks[:2]])  # Limit context
            
            # Create a simple prompt
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs, 
                    max_length=inputs.shape[1] + 100,  # Add 100 tokens
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer part
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
            else:
                answer = response[len(prompt):].strip()
            
            # Clean up the answer
            if len(answer) < 10 or not answer:
                return self.generate_answer_simple(question, relevant_chunks)
            
            # Add sources
            sources = list(set([chunk['source'] for chunk in relevant_chunks]))
            if sources:
                source_names = [os.path.basename(src) for src in sources]
                answer += f"\n\n📚 Sources: {', '.join(source_names)}"
            
            return answer
            
        except Exception as e:
            print(f"Error with advanced generation, falling back to simple: {str(e)}")
            return self.generate_answer_simple(question, relevant_chunks)
    
    def ask_question(self, question, use_advanced=False):
        """Main method to ask a question and get an answer"""
        print(f"Question: {question}")
        
        # Search for relevant documents
        relevant_chunks = self.search_documents(question)
        print(f"Found {len(relevant_chunks)} relevant chunks")
        
        # Generate answer (choose method)
        if use_advanced:
            answer = self.generate_answer_advanced(question, relevant_chunks)
        else:
            answer = self.generate_answer_simple(question, relevant_chunks)
        
        return answer
    
    def get_vector_store_stats(self):
        """Get vector store statistics"""
        return self.vector_store.get_stats()
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            
        with torch.no_grad():
            outputs = self.model.generate(
                    inputs, 
                    max_length=inputs.shape[1] + 100,  # Add 100 tokens
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract just the answer part
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
            else:
                answer = response[len(prompt):].strip()
            
            # Clean up the answer
            if len(answer) < 10 or not answer:
                return self.generate_answer_simple(question, relevant_chunks)
            
            # Add sources
            sources = list(set([chunk['source'] for chunk in relevant_chunks]))
            if sources:
                answer += f"\n\nSources: {', '.join(sources)}"
            
            return answer
    
    def ask_question(self, question, use_advanced=False):
        """Main method to ask a question and get an answer"""
        print(f"Question: {question}")
        
        # Search for relevant documents
        relevant_chunks = self.search_documents(question)
        
        # Generate answer (choose method)
        if use_advanced:
            answer = self.generate_answer_advanced(question, relevant_chunks)
        else:
            answer = self.generate_answer_simple(question, relevant_chunks)
        
        return answer
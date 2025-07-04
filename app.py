import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import io
import re
from typing import List, Dict, Tuple
import json

# Configure page
st.set_page_config(
    page_title="CV Screening Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

class CVScreeningRAG:
    def __init__(self, gemini_api_key: str):
        """Initialize the RAG system with Gemini API"""
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Initialize sentence transformer for embeddings
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            st.error(f"Error loading sentence transformer: {e}")
            self.embedder = None
        
        self.cv_data = []
        self.cv_embeddings = None
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            pdf_reader = PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def process_cvs(self, uploaded_files) -> None:
        """Process uploaded CV files and create embeddings"""
        self.cv_data = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f'Processing {uploaded_file.name}...')
            
            # Extract text from PDF
            cv_text = self.extract_text_from_pdf(uploaded_file)
            
            if cv_text:
                cleaned_text = self.clean_text(cv_text)
                
                # Store CV data
                self.cv_data.append({
                    'filename': uploaded_file.name,
                    'text': cleaned_text,
                    'candidate_name': self.extract_candidate_name(cleaned_text)
                })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Create embeddings for all CVs
        if self.cv_data and self.embedder:
            status_text.text('Creating embeddings...')
            cv_texts = [cv['text'] for cv in self.cv_data]
            self.cv_embeddings = self.embedder.encode(cv_texts)
        
        status_text.text('✅ Processing complete!')
        progress_bar.empty()
        status_text.empty()
    
    def extract_candidate_name(self, cv_text: str) -> str:
        """Extract candidate name from CV text"""
        lines = cv_text.split('\n')[:5]  # Check first 5 lines
        
        # Look for patterns that might be names
        name_patterns = [
            r'^([A-Z][a-z]+ [A-Z][a-z]+)',  # First Last
            r'^([A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+)',  # First Middle Last
        ]
        
        for line in lines:
            line = line.strip()
            if len(line) > 3 and len(line) < 50:
                for pattern in name_patterns:
                    match = re.match(pattern, line)
                    if match:
                        return match.group(1)
        
        return "Unknown Candidate"
    
    def create_job_embedding(self, job_description: str):
        """Create embedding for job description"""
        if self.embedder:
            return self.embedder.encode([job_description])
        return None
    
    def retrieve_relevant_cvs(self, job_description: str, top_k: int = 5) -> List[Dict]:
        """Retrieve most relevant CVs based on job description"""
        if not self.cv_data or self.cv_embeddings is None:
            return []
        
        # Create job embedding
        job_embedding = self.create_job_embedding(job_description)
        
        if job_embedding is None:
            return []
        
        # Calculate similarities
        similarities = cosine_similarity(job_embedding, self.cv_embeddings)[0]
        
        # Get top candidates
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        relevant_cvs = []
        for idx in top_indices:
            cv_data = self.cv_data[idx].copy()
            cv_data['similarity_score'] = float(similarities[idx])
            relevant_cvs.append(cv_data)
        
        return relevant_cvs
    
    def analyze_candidate_fit(self, cv_text: str, job_description: str) -> str:
        """Use Gemini to analyze candidate fit"""
        prompt = f"""
        As an expert HR recruiter, analyze how well this candidate fits the job requirements.
        
        Job Description:
        {job_description}
        
        Candidate CV:
        {cv_text[:3000]}  # Limit text to avoid token limits
        
        Please provide:
        1. Overall fit score (1-10)
        2. Key strengths that match the role
        3. Potential gaps or concerns
        4. Specific examples from the CV that demonstrate relevant experience
        5. Recommendation (Highly Recommended/Recommended/Consider/Not Recommended)
        
        Keep your analysis concise but thorough.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error analyzing candidate: {e}"
    
    def rank_candidates(self, job_description: str, top_k: int = 10) -> List[Dict]:
        """Rank candidates and provide detailed analysis"""
        if not self.cv_data:
            return []
        
        # Retrieve relevant CVs
        relevant_cvs = self.retrieve_relevant_cvs(job_description, top_k)
        
        # Analyze each candidate with Gemini
        analyzed_candidates = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, cv in enumerate(relevant_cvs):
            status_text.text(f'Analyzing {cv["candidate_name"]}...')
            
            analysis = self.analyze_candidate_fit(cv['text'], job_description)
            
            cv['analysis'] = analysis
            cv['rank'] = i + 1
            analyzed_candidates.append(cv)
            
            progress_bar.progress((i + 1) / len(relevant_cvs))
        
        progress_bar.empty()
        status_text.empty()
        
        return analyzed_candidates

def main():
    st.title("🎯 AI-Powered CV Screening Assistant")
    st.markdown("Upload CVs and job descriptions to find the best candidates using RAG + Google Gemini")
    
    # Sidebar for API key
    with st.sidebar:
        st.header("⚙️ Configuration")
        gemini_api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key"
        )
        
        if not gemini_api_key:
            st.warning("Please enter your Gemini API key to continue")
            st.info("Get your API key from: https://makersuite.google.com/app/apikey")
            return
        
        st.success("✅ API Key configured")
        
        # Initialize RAG system
        if 'rag_system' not in st.session_state:
            st.session_state.rag_system = CVScreeningRAG(gemini_api_key)
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📤 Upload CVs", "📋 Job Requirements", "🏆 Results"])
    
    with tab1:
        st.header("Upload Candidate CVs")
        
        # Display system limits
        st.info("""
        📊 **System Limits:**
        - **Maximum CVs**: ~100-200 CVs (depending on file size)
        - **File size limit**: 200MB per file
        - **Recommended**: Start with 20-50 CVs for optimal performance
        - **API limit**: 15 requests/minute (free tier)
        """)
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload multiple CV files in PDF format"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} files uploaded")
            
            if st.button("🔄 Process CVs", type="primary"):
                with st.spinner("Processing CVs..."):
                    st.session_state.rag_system.process_cvs(uploaded_files)
                st.success(f"✅ Processed {len(uploaded_files)} CVs successfully!")
    
    with tab2:
        st.header("Job Requirements")
        
        input_method = st.radio(
            "How would you like to specify the job requirements?",
            ["📝 Write Job Description", "💼 Upload Job Description"]
        )
        
        job_description = ""
        
        if input_method == "📝 Write Job Description":
            job_description = st.text_area(
                "Job Description",
                height=200,
                placeholder="Enter the job description, required skills, qualifications, and any specific requirements..."
            )
        else:
            uploaded_jd = st.file_uploader(
                "Upload Job Description (PDF)",
                type="pdf",
                help="Upload job description as PDF file"
            )
            
            if uploaded_jd:
                job_description = st.session_state.rag_system.extract_text_from_pdf(uploaded_jd)
                st.text_area("Extracted Job Description", job_description, height=200, disabled=True)
        
        # Number of candidates to show
        if hasattr(st.session_state.rag_system, 'cv_data') and st.session_state.rag_system.cv_data:
            max_candidates = len(st.session_state.rag_system.cv_data)
            num_candidates = st.slider(
                f"Number of candidates to analyze (Max: {max_candidates})", 
                1, 
                max_candidates, 
                min(5, max_candidates)
            )
        else:
            st.info("📤 Please upload and process CVs first to set the analysis range")
            num_candidates = st.slider("Number of candidates to analyze", 1, 20, 5, disabled=True)
        
        if job_description and st.button("🎯 Find Best Candidates", type="primary"):
            if not hasattr(st.session_state.rag_system, 'cv_data') or not st.session_state.rag_system.cv_data:
                st.error("❌ Please upload and process CVs first!")
            else:
                with st.spinner("Analyzing candidates..."):
                    results = st.session_state.rag_system.rank_candidates(job_description, num_candidates)
                    st.session_state.results = results
                    st.session_state.job_description = job_description
                st.success("✅ Analysis complete! Check the Results tab.")
    
    with tab3:
        st.header("📊 Candidate Rankings")
        
        if 'results' in st.session_state and st.session_state.results:
            results = st.session_state.results
            
            # Summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Candidates", len(st.session_state.rag_system.cv_data))
            with col2:
                st.metric("Analyzed", len(results))
            with col3:
                avg_similarity = np.mean([r['similarity_score'] for r in results])
                st.metric("Avg Similarity", f"{avg_similarity:.2f}")
            
            st.divider()
            
            # Display results
            for result in results:
                with st.expander(f"🏅 #{result['rank']} - {result['candidate_name']} (Similarity: {result['similarity_score']:.3f})"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.info(f"**File:** {result['filename']}")
                        st.info(f"**Similarity Score:** {result['similarity_score']:.3f}")
                    
                    with col2:
                        st.markdown("**AI Analysis:**")
                        st.markdown(result['analysis'])
            
            # Download results
            if st.button("📥 Download Results as CSV"):
                df_results = pd.DataFrame([
                    {
                        'Rank': r['rank'],
                        'Candidate': r['candidate_name'],
                        'Filename': r['filename'],
                        'Similarity_Score': r['similarity_score'],
                        'Analysis': r['analysis']
                    }
                    for r in results
                ])
                
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="candidate_rankings.csv",
                    mime="text/csv"
                )
        
        else:
            st.info("🔍 No results yet. Please upload CVs and specify job requirements first.")

if __name__ == "__main__":
    main()
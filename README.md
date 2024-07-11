# PaperNexus : Smart Research Repository
PaperNexus is a smart research assistant application that leverages advanced techniques from the multisource agent of the RAG (Retrieval-Augmented Generation) pipeline. It enables users to efficiently query your research papers in PDF documents and Arxiv papers.

## Features
- **PDF Document Queries:** Extracts and analyzes information from PDF documents using advanced embeddings and vector stores.
- **Arxiv Paper Queries:** Utilizes ArxivAPIWrapper to perform sophisticated queries on Arxiv papers, providing insightful results.
- **Intelligent Assistance:** Communicate naturally with the assistant using open source llm models `llama3-8b-8192` from Groq Inference.
- **Integration:** Integrates tools like CohereEmbeddings and FAISS for efficient document embedding and retrieval.

## Technology Used
- Python 3.10
- Streamlit
- Groq Inference : Open-Source LLM Models
- Langchain
- CohereEmbeddings and FAISS Vector Stores

## Installation
1. **Clone the repository:**
   ``` bash
   git clone https://github.com/aashish1008/PaperNexus.git
   cd PaperNexus
2. **Install dependencies:**
   ``` bash
   pip install -r requirements.txt

3. **Set up environment variables:**
   - Obtain API keys for Cohere and GROQ.
   - Create a `.env` file in the root directory:
     ``` bash
     COHERE_API_KEY=your_cohere_api_key
     GROQ_API_KEY=your_groq_api_key

## Usage
1. Run the application:
   ``` bash
   streamlit run app.py
2. Open your web browser and navigate to http://localhost:8501.
3. Interact with PaperNexus by asking questions related to PDF documents or Arxiv papers.

   

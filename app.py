import streamlit as st
import os
import time

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper


class PaperNexus:
    def __init__(self):
        self.cohere_api_key, self.groq_api_key = self.setup_env()
        self.pdf_files = "attention.pdf"  # Initialize PDF file (you can expand this to handle multiple PDF files if needed)
        self.vectordb = self.pdf_loader()
        self.tools = [self.create_pdf_retriever_tool(), self.create_arxiv_tool()]


    def setup_env(self):
        # Load environment variables
        load_dotenv()

        # Set environment variables
        cohere_api_key = os.getenv("COHERE_API_KEY")
        groq_api_key = os.getenv('GROQ_API_KEY')

        return cohere_api_key, groq_api_key

    def pdf_loader(self):
        embeddings = CohereEmbeddings(cohere_api_key=self.cohere_api_key)
        loader = PyPDFLoader(self.pdf_files)
        docs = loader.load()
        vector_db = FAISS.from_documents(docs, embeddings)
        return vector_db

    def create_arxiv_tool(self):
        # Initialize components for Arxiv queries
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=5000)
        arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

        return arxiv

    def create_pdf_retriever_tool(self):
        # Create retriever tool for PDF files
        retriever_pdf = self.vectordb.as_retriever()
        pdf_retriever_tool = create_retriever_tool(retriever_pdf, "PDF Files",
                                                   "Search for information from PDF files. Please provide specific queries related to these documents.")
        return pdf_retriever_tool

    def setup_llm_prompts(self):
        # Initialize ChatGroq instance
        llm = ChatGroq(groq_api_key=self.groq_api_key,
                       model_name="llama3-8b-8192"
                       )

        # Define ChatPromptTemplate with enhanced prompt structure
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Welcome to our PDF and Arxiv paper query assistant! How can I assist you today with your PDF documents or Arxiv paper searches? If your query falls outside these areas, I'm afraid I won't be able to assist you, but I'm here to help with anything related to PDFs or Arxiv papers."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
            ("placeholder", "{page_content}")
        ])

        return llm, prompt

    def run(self):
        llm, prompt = self.setup_llm_prompts()

        # Create agent with improved handling and execution
        agent = create_tool_calling_agent(llm, self.tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)

        # Streamlit UI
        st.set_page_config(page_title="PaperNexus")

        st.markdown("<h1 style='text-align: center;'>PaperNexus</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Smart Research Repository</h3>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center;'>Query your Research Papers and Arxiv Papers effortlessly.</h4><br>",
                    unsafe_allow_html=True)

        # Input prompt from user
        input_prompt = st.text_input("Ask a question about the document (PDF or Arxiv)")

        # Handle prompt input
        if input_prompt:
            start = time.process_time()
            response = agent_executor.invoke({
                "input": input_prompt
            })

            # Calculate and display response time
            st.write("Response time :", time.process_time() - start)
            st.write("Assistant's Response:")
            st.write(response['output'])


# Run the app
if __name__ == "__main__":
    SRR = PaperNexus()
    SRR.run()

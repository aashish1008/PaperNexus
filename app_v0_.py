import streamlit as st
import os
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings  # Fixed import
from langchain_community.llms import CTransformers
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper
from PyPDF2 import PdfReader


class VectorDB:
    def __init__(self, txts, md, embeddings):
        self.txts = txts
        self.metadatas = md
        self.embeddings = embeddings
        self.vectorDBLoc = "local_db/nexus_index"

    def setup_local_db(self):
        # Check if local DB already exists
        if os.path.exists(self.vectorDBLoc):
            return FAISS.load_local(self.vectorDBLoc, self.embeddings, allow_dangerous_deserialization=True)

        # If not, create a new one
        db = FAISS.from_texts(texts=self.txts, embedding=self.embeddings, metadatas=self.metadatas)
        db.save_local(self.vectorDBLoc)
        return db


class PaperNexus:
    def __init__(self):
        self.txts = None
        self.metadatas = None

    def extract_info_from_multiple_pdf(self, upl_files):
        self.txts = []
        self.metadatas = []

        txt_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        for file in upl_files:
            pdf_reader = PdfReader(file)

            file_texts = []
            for page in pdf_reader.pages:
                file_texts.append(page.extract_text())

            text = "\n".join(file_texts)

            # Split the text into chunks
            texts = txt_splitter.split_text(text)
            self.txts.extend(texts)

            # Create metadata for each chunk
            metadatas = [{"source": f"{file.name}-{j}"} for j in range(len(texts))]
            self.metadatas.extend(metadatas)

    def load_local_vectorDB(self):
        model_name = "sentence-transformers/all-mpnet-base-v2"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(
                    model_name=model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
        )
        vectorDB = VectorDB(txts=self.txts, md=self.metadatas, embeddings=embeddings)
        return vectorDB.setup_local_db()

    def create_arxiv_tool(self):
        # Initialize components for Arxiv queries
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=5000)
        arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

        return arxiv

    def create_pdf_retriever_tool(self):
        # Create retriever tool for PDF files
        retriever_pdf = self.load_local_vectorDB().as_retriever()
        pdf_retriever_tool = create_retriever_tool(retriever_pdf, "PDF Files",
                                                   "Search for information from PDF files. Please provide specific queries related to these documents.")
        return pdf_retriever_tool

    def setup_llm(self):
        # Initialize ChatGroq instance
        llm_model = CTransformers(
            model="llama-2-7b-chat.ggmlv3.q4_1.bin",
            model_type="llama",
            max_new_tokens=512,
            temperature=0.5
        )

        return llm_model

    def custom_prompt(self):
        # Define ChatPromptTemplate with enhanced prompt structure
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Welcome to our PDF and Arxiv paper query assistant! How can I assist you today with your PDF documents or Arxiv paper searches? If your query falls outside these areas, I'm afraid I won't be able to assist you, but I'm here to help with anything related to PDFs or Arxiv papers."),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
            ("placeholder", "{page_content}")
        ])

        return prompt

    def run(self):
        # Streamlit UI
        st.set_page_config(page_title="PaperNexus")

        st.markdown("<h1 style='text-align: center;'>PaperNexus</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align: center;'>Smart Research Repository</h3>", unsafe_allow_html=True)
        st.markdown(
            "<h4 style='text-align: center;'>Query your Research Papers and Arxiv Papers effortlessly.</h4><br>",
            unsafe_allow_html=True)
        files_path = st.file_uploader("Choose multiple PDF files", accept_multiple_files=True)

        if files_path:
            self.extract_info_from_multiple_pdf(files_path)
            # Create tools
            tools = [self.create_pdf_retriever_tool(), self.create_arxiv_tool()]
            # Create agent with improved handling and execution
            agent = create_tool_calling_agent(self.setup_llm(), tools, self.custom_prompt())
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

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

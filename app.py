import os
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS  # Vectorstore DB
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Vector generation technique
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

st.title("📄 LAAMA Model Document Q & A")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")  # Removed space in model name

# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only.
    Please provide the most accurate response based on the question.
    
    <context>
    {context}
    </context>
    
    Question: {input}
    """
)

# Function to generate vector embeddings
def vector_embedding():
    try:
        st.info("🔄 Generating vector embeddings... Please wait.")

        # Initialize Embedding Model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Load PDFs from the folder
        pdf_folder = "./us_census"
        docs = []

        if os.path.exists(pdf_folder) and os.listdir(pdf_folder):
            for filename in os.listdir(pdf_folder):
                if filename.endswith(".pdf"):
                    loader = PyPDFLoader(os.path.join(pdf_folder, filename))
                    docs.extend(loader.load())

            if not docs:
                st.error("⚠️ No PDF content found! Please check your files.")
                return

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)

            # Create FAISS Vector Store
            st.session_state.vectors = FAISS.from_documents(final_documents, embeddings)
            st.session_state.vector_store_ready = True  # Flag to indicate vector store is ready
            st.session_state.docs = docs  # Store documents for reference

            st.success("✅ Vector Store is Ready!")
        else:
            st.error("⚠️ No PDFs found in `us_census/` folder! Please add files and try again.")
    except Exception as e:
        st.error(f"⚠️ Error in vector embedding: {e}")

# Button to create vector store
if st.button("🚀 Create Vector Store"):
    vector_embedding()

# Ensure session persistence
if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = False

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []  # Stores history of Q&A pairs

# User input for questions
prompt1 = st.text_input("💡 What do you want to ask from the document?")

if prompt1:
    if not st.session_state.vector_store_ready:
        st.error("⚠️ Please create the vector store first by clicking the button above.")
    else:
        # Create Document Retrieval Chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)

        with st.spinner("🔍 Searching for the best answer..."):
            response = retriever_chain.invoke({'input': prompt1})

        # Get Answer
        answer = response.get('answer', "❌ No valid response received.")

        # Store Q&A in session state
        st.session_state.qa_history.append({"question": prompt1, "answer": answer})

        # Display Latest Q&A
        st.subheader("💡 Latest Answer:")
        st.write(f"**❓ Question:** {prompt1}")
        st.write(f"✅ **Answer:** {answer}")

# Display Q&A History in an Expandable Section
with st.expander("📝 View Q&A History"):
    for qa in reversed(st.session_state.qa_history[:-1]):  # Exclude latest
        st.write(f"**❓ Question:** {qa['question']}")
        st.write(f"✅ **Answer:** {qa['answer']}")
        st.write("---")  # Separator between Q&A entries

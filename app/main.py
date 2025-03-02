import streamlit as st  # Web app framework
import os  # File path operations
import requests  # Hugging Face API requests
from langchain.document_loaders import PyPDFLoader  # PDF Loader for extracting text
from langchain.text_splitter import CharacterTextSplitter  # Splits text into chunks
from langchain.embeddings import OpenAIEmbeddings  # You may need to replace this later
from langchain.vectorstores import Chroma  # Chroma Vector Store for retrieval
import chromadb  # ChromaDB for vector storage
from dotenv import load_dotenv  # Load environment variables

# --- Load API keys ---
load_dotenv()
hf_api_key = st.secrets["HF_API_KEY"]  # Use Hugging Face API key

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="MWC25 AI Chatbot", page_icon="🤖", layout="wide")

# --- Header with Logo & Title ---
st.image("../images/logo.jpg", width=180)
st.markdown("""
    <div class='title-container' style='text-align: center;'>
        <h1>🤖 MWC25 AI Chatbot</h1>
        <p>Your AI-powered assistant for technical insights</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""<br><br>""", unsafe_allow_html=True)  # Spacer

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Display Chat History ---
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input ---
user_question = st.chat_input("Ask me anything...")

if user_question:
    st.session_state["messages"].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.markdown(user_question)

    # --- Processing Animation ---
    with st.spinner("Thinking... 🤔"):
        document_dir = "./"
        pdf_file = "../data/AI-Cytology.pdf"

        @st.cache_resource
        def load_and_embed_document():
            """Loads the PDF, splits it into chunks, and creates vector embeddings."""
            file_path = os.path.join(document_dir, pdf_file)
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
            chunks = text_splitter.split_documents(pages)
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # Consider replacing this
            client = chromadb.PersistentClient(path="./chroma_db")
            db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db", client=client)
            return db

        db = load_and_embed_document()
        retrieved_docs = db.similarity_search(user_question, k=10)

        if retrieved_docs:
            source_info = "\n".join(
                [f"📖 **Source:** {os.path.basename(doc.metadata.get('source', 'Unknown'))}, Page: {doc.metadata.get('page', 'N/A')}" for doc in retrieved_docs]
            )
            context_text = retrieved_docs[0].page_content
        else:
            source_info = "❌ **No relevant information found.**"
            context_text = "No context available."

        # --- Construct AI Prompt ---
        prompt = f"""
        ## SYSTEM ROLE
        You are an AI assistant providing concise, accurate answers based only on the given context.

        ## USER QUESTION
        "{user_question}"

        ## CONTEXT
        '''
        {context_text}
        '''

        ## RESPONSE FORMAT
        **Answer:** [Concise response]

        📌 **Key Insights:**
        - Bullet point 1
        - Bullet point 2

        {source_info}
        """

        # --- Call Hugging Face API ---
        def query_huggingface(text):
            api_url = "https://api-inference.huggingface.co/models/bigscience/bloom"  # Change model if needed
            headers = {"Authorization": f"Bearer {hf_api_key}"}
            payload = {"inputs": text, "parameters": {"max_length": 400}}

            response = requests.post(api_url, headers=headers, json=payload)
            return response.json()

        response = query_huggingface(prompt)

        if isinstance(response, list) and "generated_text" in response[0]:
            answer = response[0]['generated_text']
        else:
            answer = "⚠️ Error: Unable to fetch response from Hugging Face API."

    # --- Display AI Response ---
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
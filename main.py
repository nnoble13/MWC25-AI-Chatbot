import streamlit as st  # Web app framework
import os  # File path operations
import openai  # OpenAI API for generating responses
from langchain.document_loaders import PyPDFLoader  # PDF Loader for extracting text
from langchain.text_splitter import CharacterTextSplitter  # Splits text into chunks
from langchain.embeddings import OpenAIEmbeddings  # Embedding model for vector search
from langchain.vectorstores import Chroma  # Chroma Vector Store for retrieval
import chromadb  # ChromaDB for vector storage
from dotenv import load_dotenv  # Load environment variables

# --- Load API keys ---
load_dotenv()
# api_key = os.getenv("OPENAI_API_KEY") # to run it locally
api_key = st.secrets["OPENAI_API_KEY"] # to run it on streamlit
openai.api_key = api_key

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
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
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

        # --- Call OpenAI GPT ---
        client = openai.OpenAI()
        model_params = {
            'model': 'gpt-4o',
            'temperature': 0.7,
            'max_tokens': 4000,
            'top_p': 0.9,
            'frequency_penalty': 0.5,
            'presence_penalty': 0.6
        }

        messages = [{'role': 'user', 'content': prompt}]
        completion = client.chat.completions.create(messages=messages, **model_params, timeout=120)
        answer = completion.choices[0].message.content

    # --- Display AI Response ---
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
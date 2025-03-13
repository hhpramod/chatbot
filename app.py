import streamlit as st
import google.generativeai as genai
import faiss
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import datetime
import time
import webbrowser


# Load FAISS index and Sentence Transformer model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index")  # Load pre-created FAISS index

# Load text chunks (ensure they are in the same order as when indexed)
def load_chunks():
    with open("text_chunks.txt", "r", encoding="utf-8") as f:
        return f.read().split("\n\n")

chunks = load_chunks()

# Configure Gemini API
genai.configure(api_key="AIzaSyD8EmwPwsLmCgiTyKUcl_z-MNtvRD1TRlg")

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query):
    query_embedding = model.encode([query])
    _, indices = index.search(query_embedding, k=3)  # Get top 3 results
    return [chunks[i] for i in indices[0]]

# Function to get a time-based greeting
def get_greeting():
    current_hour = datetime.datetime.now().hour
    if current_hour < 12:
        return "Good morning! ☀️"
    elif current_hour < 18:
        return "Good afternoon! 🌤️"
    else:
        return "Good evening! 🌙"

# Chatbot function
def chatbot(query):
    if query.lower() in ["hi", "hello", "hey", "hey there", "hi there", "hello there"]:
        return "Hello! How can I assist you today? 😊"
    elif query.lower() in ["bye", "goodbye", "thank you", "thanks", "thanks a lot", "thank you very much"]:
        return "You're welcome! Have a great day! 👋 "

    relevant_chunks = retrieve_relevant_chunks(query)
    if not relevant_chunks:
        webbrowser.open(f"https://www.google.com/search?q={query.replace(' ', '+')}")
        return "I'm not sure about that, but I searched the web for you. Check the results in your browser."

    relevant_text = "\n\n".join(relevant_chunks)
    prompt = f"""
    You are an expert AI assistant trained on Diabetes Context.
    
    Context:
    {relevant_text}
    
    Answer the following question related to diabetes with basic idea. After that if user ask more information about the answer, give more detailed information as much as possible.
    Use general medical knowledge or web search content where needed to answer the questions.
    At the end of each chat ask the user to ask more questions based on the answer.
    
    User: {query}
    Assistant:
    """
    
    try:
        model_gemini = genai.GenerativeModel("gemini-2.0-flash")
        response = model_gemini.generate_content(prompt)
        return response.text if response.text else "I'm not sure about that, but I searched the web for you."
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit Chatbot UI
st.set_page_config(page_title="AI Medical Chatbot", page_icon="👨‍⚕️")

# Sidebar for chat history
st.sidebar.title("Chat History & Web Search")
if "chat_sessions" not in st.session_state:
    st.session_state["chat_sessions"] = {}
    st.session_state["current_chat"] = "Session 1"
    st.session_state["messages"] = []

# Select or create new chat session
session_names = list(st.session_state["chat_sessions"].keys())
new_session = st.sidebar.text_input("Start new session:")
if st.sidebar.button("Create") and new_session:
    if new_session not in session_names:
        st.session_state["chat_sessions"][new_session] = []
        st.session_state["current_chat"] = new_session
    else:
        st.sidebar.warning("Session name already exists.")

# Select session from dropdown
if session_names:
    selected_session = st.sidebar.selectbox("Select session:", session_names, index=session_names.index(st.session_state["current_chat"]))
    st.session_state["current_chat"] = selected_session
else:
    selected_session = "Session 1"

# Ensure selected session has message storage
if selected_session not in st.session_state["chat_sessions"]:
    st.session_state["chat_sessions"][selected_session] = []

# Web Search Option
st.sidebar.subheader("Search on the Web")
search_query = st.sidebar.text_input("Enter search query:")
if st.sidebar.button("Search") and search_query:
    search_url = f"https://www.google.com/search?q={search_query.replace(' ', '+')}"
    webbrowser.open(search_url)

# Chatbot details in sidebar with different font
st.sidebar.subheader("About This Chatbot")
st.sidebar.markdown(
    """
    <div style="font-family:Courier; font-size:14px;">
    <b>AI Medical Chatbot</b> is designed to assist users with diabetes-related inquiries. 
    It provides detailed answers using expert medical knowledge and web searches when needed.
    </div>
    """,
    unsafe_allow_html=True
)


# Load chat messages from session
st.session_state["messages"] = st.session_state["chat_sessions"][selected_session]

# Display animated greeting
st.markdown(f"""<h1 style='text-align: center;'>{get_greeting()}</h1>""", unsafe_allow_html=True)
time.sleep(1)  # Simulate animation effect
st.markdown("""<h2 style='text-align: center;'>👨‍⚕️ AI Medical Chatbot</h2>""", unsafe_allow_html=True)

# Display chat history
for message in st.session_state["messages"]:
    role, text = message["role"], message["text"]
    with st.chat_message(role):
        st.markdown(text)

# Input box for user query
user_input = st.chat_input("Type your message...")
if user_input:
    # Add user message to chat history
    st.session_state["messages"].append({"role": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate bot response
    with st.spinner("Thinking... 🤔"):
        response = chatbot(user_input)
    
    # Animate bot response like typing
    with st.chat_message("assistant"):
        response_container = st.empty()
        typed_response = ""
        for char in response:
            typed_response += char
            response_container.markdown(typed_response)
            time.sleep(0.02)  # Simulate typing speed
    
    # Add bot response to chat history
    st.session_state["messages"].append({"role": "assistant", "text": response})
    
    # Save chat history to the selected session
    st.session_state["chat_sessions"][selected_session] = st.session_state["messages"]
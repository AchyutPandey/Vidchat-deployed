"""
ChatYT Streamlit App (LCEL Chain Version)
This Streamlit app enables you to:
* Summarise YouTube videos
* Ask questions about the topics discussed in the video
It uses LangChain Expression Language (LCEL) with Google's Gemini APIs.
"""

import streamlit as st
import yt_dlp
import os
# Corrected import: Document is now in langchain_core.documents
from langchain_core.documents import Document
# Corrected import: RecursiveCharacterTextSplitter is in its own package
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# Corrected import: ChatPromptTemplate is now in langchain_core.prompts
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import google.generativeai as genai
import time

# --- App Configuration ---
st.set_page_config(
    page_title="ChatYT (LangChain)",
    page_icon="📺",
    layout="wide",
)

st.title("📺 ChatYT: Chat with any YouTube Video")
st.caption("Summarize and ask questions about any YouTube video using LangChain and Google Gemini.")

# --- API Key Handling ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


if not GEMINI_API_KEY:
    GEMINI_API_KEY = st.sidebar.text_input(
        "Enter your Gemini API Key:", type="password"
    )

if not GEMINI_API_KEY:
    st.error("Please provide your Gemini API Key in the sidebar to continue.")
    st.stop()

# Configure the genai library (still needed for file upload)
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}")
    st.stop()


# --- Core Functions (Caching to improve performance) ---

@st.cache_data(show_spinner="Downloading audio...")
def download_audio(link, file_name='audio.mp3'):
    """
    Downloads the audio from a YouTube link.
    """
    with yt_dlp.YoutubeDL({'extract_audio': True,
                           'format': 'worstaudio',
                           'overwrites': True,
                           'outtmpl': file_name}) as video:
        info_dict = video.extract_info(link, download=True)
        video_title = info_dict['title']
    return file_name, video_title

@st.cache_data(show_spinner="Compressing audio...")
def compress_audio(input_file, output_file="compressed.mp3"):
    """
    Compresses the audio file for faster API uploads.
    """
    os.system(f"ffmpeg -y -i {input_file} -ar 16000 -ac 1 {output_file}")
    return output_file

@st.cache_data(show_spinner="Transcribing video...")
def speech_to_text(audio_file):
    """
    Transcribes audio using the Gemini API.
    (This function uses the base genai library for file upload)
    """
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        audio_file_uploaded = genai.upload_file(path=audio_file, mime_type="audio/mpeg")
        
        prompt = "Please transcribe this audio file. Provide only the text transcription."
        response = model.generate_content([prompt, audio_file_uploaded])
        
        genai.delete_file(audio_file_uploaded.name)
        
        if response.candidates and response.candidates[0].content.parts:
            return response.candidates[0].content.parts[0].text
        else:
            return f"Error: Could not transcribe audio. Response: {response}"
    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")
        try:
            if 'audio_file_uploaded' in locals():
                genai.delete_file(audio_file_uploaded.name)
        except Exception as e_del:
            st.warning(f"Error cleaning up file: {e_del}")
        return f"Error: {e}"

@st.cache_data(show_spinner="Summarizing text...")
def summarize_text_api(text):
    """
    Summarizes the text using a LangChain chain.
    """
    # 1. Define the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                               temperature=0.3,
                               google_api_key=GEMINI_API_KEY)
    
    # 2. Define the Prompt
    prompt_template = """Please provide a concise, high-level summary of the following text:
    ---
    {text}
    ---
    Provide only the summary."""
    summarize_prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # 3. Define the Chain
    summarize_chain = summarize_prompt | llm | StrOutputParser()
    
    try:
        # 4. Invoke the Chain
        response = summarize_chain.invoke({"text": text})
        return response
    except Exception as e:
        st.error(f"An error occurred during summarization: {e}")
        return f"Error: {e}"

@st.cache_data(show_spinner="Generating embeddings...")
def generate_embeddings_db(text):
    """
    Splits text, generates embeddings via API, and stores in ChromaDB.
    Returns the Chroma database object.
    """
    doc = Document(page_content=text, metadata={"source": "youtube"})
    # This now uses the imported RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents([doc])
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                                  google_api_key=GEMINI_API_KEY)
        db = Chroma.from_documents(chunks, embeddings)
        return db
    except Exception as e:
        st.error(f"An error occurred during embedding generation: {e}")
        return None

def format_docs(docs):
    """Helper function to format retrieved documents into a string."""
    if not docs:
        return "No relevant context found."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# --- Streamlit UI Components ---

# Initialize session state variables
if "summary" not in st.session_state:
    st.session_state.summary = ""
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "video_title" not in st.session_state:
    st.session_state.video_title = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

url = st.text_input("Enter YouTube URL:", key="youtube_url")

if st.button("Process Video", key="process_video"):
    if url:
        with st.spinner("Processing video... This may take a few minutes."):
            try:
                # Reset state
                st.session_state.summary = ""
                st.session_state.rag_chain = None
                st.session_state.video_title = ""
                st.session_state.chat_history = []

                # 1. Download
                audio_file, video_title = download_audio(url)
                st.session_state.video_title = video_title
                
                # 2. Compress
                compressed_audio = compress_audio(audio_file)
                
                # 3. Transcribe
                text = speech_to_text(compressed_audio)
                if "Error:" in text:
                    st.error(f"Failed to transcribe: {text}")
                    st.stop()
                
                # 4. Summarize (using the new chain function)
                summary = summarize_text_api(text)
                st.session_state.summary = summary
                
                # 5. Embed and create DB
                db = generate_embeddings_db(text)
                
                if db:
                    # 6. Create RAG Chain and store it in session state
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                                               temperature=0.3,
                                               google_api_key=GEMINI_API_KEY)
                    
                    retriever = db.as_retriever(search_kwargs={"k": 3})
                    
                    PROMPT_TEMPLATE = """Answer the following questions based only on the following context:
                    {context}
                    ---
                    Answer the question based on the above context:
                    {question}
                    """
                    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                    
                    # This is the RAG chain
                    rag_chain = (
                        {"context": retriever | format_docs, "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )
                    
                    st.session_state.rag_chain = rag_chain
                    st.success("Video processed and Q&A chat is ready!")
                else:
                    st.error("Failed to create vector database.")
                
                # Clean up local files
                try:
                    os.remove(audio_file)
                    os.remove(compressed_audio)
                except OSError as e:
                    st.warning(f"Could not clean up audio files: {e}")
                    
            except Exception as e:
                st.error(f"An error occurred during video processing: {e}")
    else:
        st.warning("Please enter a YouTube URL.")

# --- Display Summary and Chat Interface ---

if st.session_state.summary:
    st.subheader(f"Summary for: *{st.session_state.video_title}*")
    st.markdown(st.session_state.summary)
    
    st.subheader("Ask Questions")
    
    # Display chat history
    for author, message in st.session_state.chat_history:
        with st.chat_message(author):
            st.markdown(message)
            
    # Chat input
    if prompt := st.chat_input("Ask a question about the video..."):
        if st.session_state.rag_chain:
            # Add user message to history
            st.session_state.chat_history.append(("user", prompt))
            with st.chat_message("user"):
                st.markdown(prompt)
                
            # Generate and display bot response by invoking the chain
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Here we just invoke the chain with the prompt!
                    answer = st.session_state.rag_chain.invoke(prompt)
                    st.markdown(answer)
                    
            # Add bot message to history
            st.session_state.chat_history.append(("assistant", answer))
        else:
            st.error("The Q&A chain is not ready. Please process a video first.")


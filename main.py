import os
import logging
from io import BytesIO
from dotenv import load_dotenv
from gtts import gTTS
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
import streamlit as st
from streamlit_markmap import markmap
from transformers import pipeline
import torch
import wordninja
import re
import nltk
from nltk.tokenize import sent_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set Streamlit page configuration
st.set_page_config(page_title="Document Dialogue", layout="wide")

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Check if the API key is loaded
if not api_key:
    st.error("Google API key not found.")
else:
    genai.configure(api_key=api_key)

# Set logging configuration
logging.basicConfig(level=logging.INFO)


@st.cache_resource
def load_summarizer():
    try:
        device = 0 if torch.cuda.is_available() else -1
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
        logging.info(f"Summarization model loaded on device: {device}")
        return summarizer
    except Exception as e:
        logging.error(f"Error loading summarizer: {e}")
        st.error("Failed to load the summarization model.")
        return None


summarizer = load_summarizer()


def gemini_image(image):
    model = genai.GenerativeModel('gemini-1.5-pro')
    prompt = 'Generate a detailed description of the image provided.'
    try:
        chain = model.generate_content(contents=[prompt, image])
        chain.resolve()
        return chain.text
    except Exception as e:
        logging.error(f"Error generating image description: {e}")
        return "Image description unavailable."


def clean_text(text):
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)
    text = re.sub(r'([A-Za-z])(\d)', r'\1 \2', text)
    text = re.sub(r'\.(?!\s)', r'. ', text)
    text = re.sub(r',(?=\S)', r', ', text)
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = split_concatenated_words(text, min_word_length=15)

    sentences = sent_tokenize(text)
    sentences = [s.strip().capitalize() for s in sentences]
    text = ' '.join(sentences)

    return text.strip()


def getpdf(pdf_files):
    """
    Extract text from uploaded PDF files and clean the text.
    """
    text = ""
    for pdf_file in pdf_files:
        try:
            pdf_reader = PdfReader(pdf_file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_content = page.extract_text()
                if page_content:
                    cleaned_page = clean_text(page_content)
                    text += cleaned_page + "\n"
        except Exception as e:
            logging.error(f"Error reading PDF file {pdf_file.name}: {e}")
            st.error(f"Failed to read PDF file {pdf_file.name}.")
    return text


def get_chunks(data, chunk_size=1000, chunk_overlap=200):
    text_split = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_split.split_text(data)
    logging.info(f"Text split into {len(chunks)} chunks.")
    return chunks


def get_vector(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local('vector_store')
        logging.info("Vector store created and saved locally.")
    except Exception as e:
        logging.error(f"Error creating vector store: {e}")


def get_accuracy(ai_response, pdf_data):
    model = genai.GenerativeModel('gemini-1.5-pro')
    prompt = ('Give me an accuracy in percentage of how much the data are related to each other '
              'only return the percentage and the accuracy should be between 70-100% it cannot be anything else')
    try:
        chain = model.generate_content(contents=[prompt, str(ai_response), str(pdf_data)])
        chain.resolve()
        return chain.text
    except Exception as e:
        logging.error(f"Error in get_accuracy: {e}")
        return "Error calculating accuracy"


def get_conversation_chain(temp, top_k, top_p):
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
    The user wants to chat with the PDF, so help them out by ensuring the user is not disappointed with the response. 
    Analyze the context thoroughly as much as possible. Avoid speculation and focus on verifiable information.

    # Context:
    {context} 

    # Question: 
    {question} 

    # Answer: 
    """
    try:
        model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=temp, top_k=top_k, top_p=top_p)
        prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        logging.info("Conversation chain initialized successfully.")
        return chain
    except Exception as e:
        logging.error(f"Error initializing conversation chain: {e}")
        st.error("Failed to initialize the conversation chain.")
        return None


def split_concatenated_words(text, min_word_length=15):
    words = text.split()
    split_words = []
    for word in words:
        if len(word) > min_word_length:
            split = wordninja.split(word)
            split_words.extend(split)
        else:
            split_words.append(word)
    return ' '.join(split_words)


def user_input(user_question, temp, top_k, top_p):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("vector_store", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)

        if not docs:
            st.warning("No relevant documents found.")
            return

        chain = get_conversation_chain(temp, top_k, top_p)
        if not chain:
            st.error("Conversation chain is not available.")
            return

        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        st.success(response["output_text"])
    except Exception as e:
        logging.error(f"Error in user_input: {e}")
        st.error("An error occurred while processing your input. Please try again.")


def speak_text(text):
    try:
        os.makedirs("temp", exist_ok=True)
        tts = gTTS(text, lang="en")
        tts.save("temp/temp.mp3")
        audio_file = open("temp/temp.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/ogg")
        st.markdown(""" 
            <style>
            .stAudio{
                width: 300px !important;
            }</style>
            """, unsafe_allow_html=True)
    except Exception as e:
        logging.error(f"Error in speak_text: {e}")
        st.write("Error in text-to-speech.")


def generate_markdown(text):
    query = rf"""
        Study the given {text} and generate a summary then please be precise in selecting the data such that it gets to a hierarchical structure. 
        Don't give anything else, I just want to display the structure as a mindmap so be precise please. 
        Don't write anything else, Just return the md file. It is not necessary to cover all information. 
        Don't use triple backticks or ` anywhere. Cover the main topics. Please convert this data into a markdown mindmap format.
    """
    model = genai.GenerativeModel('gemini-1.5-pro')
    try:
        chain = model.generate_content(contents=[query])
        chain.resolve()
        markmap(chain.text)
    except Exception as e:
        logging.error(f"Error generating markdown: {e}")
        st.error("An error occurred while generating the mindmap.")


def generate_bart_summary(text, summarizer, max_length, min_length):
    try:
        chunks = get_chunks(text, chunk_size=1000, chunk_overlap=200)
        summaries = []

        for chunk in chunks:
            try:
                summary_result = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                summary = summary_result[0]['summary_text']
                summaries.append(summary)
            except Exception as e:
                st.error("An error occurred while generating the summary.")
                logging.error(f"Error summarizing chunk: {e}")

        final_summary = ' '.join(summaries)
        word_count = len(final_summary.split())
        st.write(f"Final Summary (Word Count: {word_count})")
        st.write(final_summary)
        st.session_state.summary = final_summary
    except Exception as e:
        logging.error(f"Error in generate_bart_summary: {e}")
        st.error("An error occurred while generating the summary.")


# Streamlit App Interface
def main():
    st.title("PDF Document Interaction")

    st.sidebar.header("Upload Your PDF")

    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

    st.sidebar.subheader("Summarizer Options")
    max_length = st.sidebar.slider("Max length", 50, 500, 200)
    min_length = st.sidebar.slider("Min length", 10, 50, 30)

    if st.sidebar.button("Summarize"):
        if uploaded_files:
            pdf_data = getpdf(uploaded_files)
            st.session_state.raw_text = pdf_data
            st.subheader("Raw Extracted Text")
            st.text_area("Text from PDF", pdf_data, height=300)

            if summarizer:
                with st.spinner("Generating summary..."):
                    generate_bart_summary(pdf_data, summarizer, max_length, min_length)
        else:
            st.warning("Please upload a PDF first.")

    st.sidebar.subheader("Conversation Parameters")
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
    top_k = st.sidebar.slider("Select Top K", min_value=1, max_value=100, value=40)
    top_p = st.sidebar.slider("Select Top P", min_value=0.0, max_value=1.0, value=0.9)

    st.subheader("Interact with the PDF")

    if st.session_state.raw_text:
        st.text_area("Extracted Text", st.session_state.raw_text, height=200)

    if st.session_state.summary:
        st.text_area("Summary", st.session_state.summary, height=200)

    user_question = st.text_input("Ask a question about the PDF")

    if st.button("Ask"):
        if user_question.strip():
            with st.spinner("Searching for an answer..."):
                user_input(user_question, temperature, top_k, top_p)
        else:
            st.warning("Please enter a question.")

    if st.button("Generate Mindmap"):
        if st.session_state.raw_text:
            with st.spinner("Generating mindmap..."):
                generate_markdown(st.session_state.raw_text)
        else:
            st.warning("No text to generate a mindmap from. Please process a PDF first.")


if __name__ == "__main__":
    main()

import streamlit as st
import langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
import fitz  # PyMuPDF
import io
import os
import re
from dotenv import load_dotenv
import glob

# Load environment variables
load_dotenv()

# Retrieve the OpenAI API key from environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')

# Function to reconstruct paragraphs from text
def reconstruct_paragraphs(text):
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

# Function to get text from a specific page of a PDF
def get_pdf_page_text(pdf_stream, page_number):
    pdf = fitz.open(stream=pdf_stream)
    page = pdf.load_page(page_number - 1)
    text = page.get_text()
    pdf.close()
    return text

# Function to read PDFs from a folder and convert them to text
def read_and_textify(pdf_folder):
    text_list = []
    sources_list = []
    file_streams = {}
    for pdf_path in glob.glob(os.path.join(pdf_folder, '*.pdf')):
        with open(pdf_path, 'rb') as f:
            file_stream = f.read()
            file_name = os.path.basename(pdf_path)
            file_streams[file_name] = file_stream
            pdf = fitz.open(stream=file_stream)
            for i in range(len(pdf)):
                text = get_pdf_page_text(io.BytesIO(file_stream), i + 1)
                text_list.append(text)
                sources_list.append(file_name + "_page_" + str(i))
            pdf.close()
    return text_list, sources_list, file_streams

# Streamlit page configuration
st.set_page_config(layout="centered", page_title="DOXS")
st.header("Imperium")
st.write("---")

# Hide Streamlit style elements
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Process all PDFs in the 'docs' folder
docs_folder = 'docs'  # Replace with your folder path
textify_output = read_and_textify(docs_folder)
documents, sources, file_streams = textify_output

# Extract embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vStore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])

# Set up the model and retriever
model_name = "gpt-4-1106-preview"
retriever = vStore.as_retriever()
retriever.search_kwargs = {'k': 2}
llm = OpenAI(model_name=model_name, openai_api_key=openai_api_key, streaming=True)
model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
retriever = vStore.as_retriever(search_type="similarity", search_kwargs={"k":1})

# Create the chain to answer questions
rqa = RetrievalQA.from_chain_type(llm=OpenAI(model_name=model_name, openai_api_key=openai_api_key, streaming=True),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)
user_q = st.text_area("Enter your questions here")

if st.button("Get Response"):
    try:
        with st.spinner("Model is working on it..."):
            result = rqa({"query": user_q})
            st.subheader('Your response:')
            st.write(result["result"])
            
            # Display the source document/part where the answer was derived from
            st.subheader('Source Document/Part:')
            st.write(result['source_documents'])


    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')

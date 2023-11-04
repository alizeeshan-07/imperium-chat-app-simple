import streamlit as st
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import RetrievalQA
import PyPDF2
from dotenv import load_dotenv
import re
import os

# WARNING: Including API keys directly in the code is not recommended for security reasons!
# Ideally, use environment variables or secure vaults to store sensitive information.


load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')

def read_and_textify(files):
    text_list = []
    sources_list = []
    for file in files:
        pdfReader = PyPDF2.PdfReader(file)
        for i in range(len(pdfReader.pages)):
            pageObj = pdfReader.pages[i]
            text = pageObj.extract_text()
            pageObj.clear()
            text_list.append(text)
            sources_list.append(file.name + "_page_"+str(i))
    return [text_list, sources_list]
  
st.set_page_config(layout="centered", page_title="Imperium")
st.header("Imperium")
st.write("---")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# file uploader in the sidebar
uploaded_files = st.sidebar.file_uploader("Upload documents", accept_multiple_files=True, type=["txt", "pdf"])

if uploaded_files is None:
    st.info(f"""Upload files to analyse""")
elif uploaded_files:
    st.write(str(len(uploaded_files)) + " document(s) loaded..")
  
    textify_output = read_and_textify(uploaded_files)
    documents = textify_output[0]
    sources = textify_output[1]
  
    # extract embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vStore = Chroma.from_texts(documents, embeddings, metadatas=[{"source": s} for s in sources])

    model_name = "gpt-4"

    retriever = vStore.as_retriever()
    retriever.search_kwargs = {'k': 4}

    llm = OpenAI(model_name=model_name, openai_api_key=openai_api_key, streaming=True)
    model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    retriever = vStore.as_retriever(search_type="similarity", search_kwargs={"k":1})

# create the chain to answer questions
    rqa = RetrievalQA.from_chain_type(llm=OpenAI(model_name=model_name, openai_api_key=openai_api_key, streaming=True),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True)
    st.header("Ask your data")
    user_q = st.text_area("Enter your questions here")
  
    if st.button("Get Response"):
        try:
            with st.spinner("Model is working on it..."):
                result = rqa({"query": user_q})
                st.subheader('Your response:')
                st.write(result["result"])
                
                # Display the source document/part where the answer was derived from
                st.subheader('Source Document/Part:')
                source_text = result['source_documents']
                st.write(source_text)
                # source_string = result['source_documents'][0].metadata.get('source', 'Unknown')
                # file_name, page = source_string.rsplit('_', 1)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error('Oops, the GPT response resulted in an error :( Please try again with a different question.')



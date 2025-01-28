import pandas as pd
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from constants import openai_key
from dotenv import load_dotenv
from langchain_openai import OpenAI

# Load environment variables
# Set OpenAI API key as environment variable
os.environ["OPENAI_API_KEY"] = openai_key

def process_data(df):
    """Process data and return processed DataFrame"""
    # Convert the data frame to text and split it into chunks
    text = df.to_string()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    
    # Get the vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    
    # Load the QA chain
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    
    # Example query
    query = "Provide a summary of the data."
    docs = vector_store.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    
    st.write("Response:")
    st.write(response)

def main():
    st.title("CSV Query Application")

    with st.sidebar:
        st.title("Work Order Processing")
        st.write("Upload a CSV file to begin")   
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")

            process_button = st.button("Process Data")
            if process_button:
                process_data(df.copy())  # Avoid modifying original data
                
        else:
           st.info("Upload a CSV file to begin.")

if __name__ == "__main__":
    main()
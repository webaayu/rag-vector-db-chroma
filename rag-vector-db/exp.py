import streamlit as st
import fitz  # PyMuPDF
from langchain.llms import Ollama
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Function to extract text from PDF file
def extract_text_from_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error occurred while reading PDF file: {e}")
        return None

# Main function
def main():
    # Set title and description
    st.title("PDF Chatbot")

    # Create a sidebar for file upload
    st.sidebar.title("Upload PDF File")
    uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=['pdf'])

    # Text input for prompt
    prompt = st.text_input("Ask a Question", "")

    if uploaded_file is not None:
        # Extract text from uploaded PDF file
        pdf_text = extract_text_from_pdf(uploaded_file)
        if pdf_text:
            # Display extracted text
            #st.subheader("Extracted Text from PDF:")
            #st.write(pdf_text)

            # Create text loader and splitter
            loader = TextLoader(pdf_text)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=20,
                length_function=len,
                is_separator_regex=False,
            )

            # Split text into documents
            texts = text_splitter.split_documents(loader.load())

            # Create embeddings
            embeddings = HuggingFaceEmbeddings()
            persist_directory = 'db1'
            vectordb = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=persist_directory)

            # Load language model for QA
            llm = Ollama(model="llama2")

            # Perform question answering
            if prompt:
                response = llm.predict(prompt, pdf_text)
                st.subheader("Generated Answer:")
                st.write(response)

if __name__ == "__main__":
    main()

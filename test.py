import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import fitz  # PyMuPDF
import docx2txt
from PIL import Image
import io
import zipfile

load_dotenv()
os.getenv("OPENAI_API_KEY")

def get_file_text(files):
    text_with_sources = []
    images_with_sources = []
    for file in files:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                if text:
                    text_with_sources.append((text, file.name, page_num + 1))
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":  # DOCX
            text = docx2txt.process(file)
            text_with_sources.append((text, file.name, 1))  # Treat as single-page
            with zipfile.ZipFile(file, "r") as docx_zip:
                for image_file in [f for f in docx_zip.namelist() if f.startswith("word/media/")]:
                    image_data = docx_zip.read(image_file)
                    images_with_sources.append((image_data, file.name))
        elif file.type == "text/plain":  # TXT
            text = file.read().decode("utf-8")
            text_with_sources.append((text, file.name, 1))  # Treat as single-page
    return text_with_sources, images_with_sources

def get_text_chunks(text_with_sources):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks_with_sources = []
    for text, doc_name, page_num in text_with_sources:
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            chunks_with_sources.append((chunk, doc_name, page_num))  # (chunk, document name, page number)
    return chunks_with_sources

def get_vector_store(text_chunks_with_sources):
    embeddings = OpenAIEmbeddings()
    texts = [chunk for chunk, _, _ in text_chunks_with_sources]
    metadatas = [{"source": doc_name, "page": page_num} for _, doc_name, page_num in text_chunks_with_sources]
    vector_store = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, files, images_with_sources):
    embeddings = OpenAIEmbeddings()
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question, k=3)
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    if docs:
        seen_sources = set()
        unique_sources = []
        for doc in docs:
            source_key = (doc.metadata["source"], doc.metadata["page"])
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                unique_sources.append({
                    "source": doc.metadata["source"],
                    "page": doc.metadata["page"]
                })

        st.write("Reply: ", response["output_text"])
        
        st.subheader("Relevant Sources:")
        for idx, source in enumerate(unique_sources, 1):
            st.write(f"{idx}. Document: {source['source']}, Page: {source['page']}")

        pdf_cache = {file.name: file.getvalue() for file in files if file.type == "application/pdf"}

        st.subheader("Related Pages & Images:")
        for source in unique_sources:
            if source["source"] in pdf_cache:
                try:
                    doc = fitz.open(stream=pdf_cache[source["source"]], filetype="pdf")
                    page = doc.load_page(source["page"] - 1)
                    pix = page.get_pixmap()
                    st.image(pix.tobytes(), 
                            caption=f"From {source['source']} (Page {source['page']})",
                            use_container_width=True)
                except Exception as e:
                    st.error(f"Couldn't load page {source['page']} from {source['source']}: {str(e)}")
            
        for image_data, source_name in images_with_sources:
            image = Image.open(io.BytesIO(image_data))
            st.image(image, caption=f"Image from {source_name}", use_container_width=True)
    else:
        st.write("No relevant document found.")

def main():
    st.set_page_config("TenderAI")
    st.header("TenderAI: Smart solutions for tender queries.")

    user_question = st.text_input("Ask a Question from the Uploaded Files")
    images_with_sources = []  # Ensure it's always initialized

    with st.sidebar:
        st.title("Menu:")
        files = st.file_uploader("Upload your PDF, DOCX, or TXT Files and Click on the Submit & Process Button", 
                                  accept_multiple_files=True, 
                                  type=["pdf", "docx", "txt"])
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text_with_sources, images_with_sources = get_file_text(files)
                text_chunks_with_sources = get_text_chunks(raw_text_with_sources)
                get_vector_store(text_chunks_with_sources)
                st.success("Done")

    if user_question and files:
        user_input(user_question, files, images_with_sources)

if __name__ == "__main__":
    main()

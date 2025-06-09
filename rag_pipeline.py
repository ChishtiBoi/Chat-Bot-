import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber

# Load environment variables
load_dotenv()

# Load and chunk the PDF
def load_and_chunk_text(pdf_path, chunk_size=1000, chunk_overlap=100):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except FileNotFoundError:
        return None, "Error: PDF file not found."

    if not text:
        return None, "No text extracted from the PDF."

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_text(text)
    print(f"Split text into {len(chunks)} chunks.")
    return chunks, None

# Create vector store
def create_vectorstore(text_chunks, embeddings):
    if not text_chunks:
        return None, "No text chunks to embed."
    db = Chroma.from_texts(text_chunks, embeddings)
    print("Embeddings computed and stored in Chroma.")
    return db, None

# Set up RetrievalQA chain with Groq
def setup_rag_chain(vectorstore):
    if vectorstore is None:
        return None, "Vector store not initialized."

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        return None, "Error: GROQ_API_KEY not set in environment variables."

    groq_llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key
    )

    prompt_template = """
    You are a business insights assistant specializing in fiscal and economic analysis. Use the provided context from the Federal Budget 2024-25 to answer the question concisely and accurately, focusing on financial metrics, policies, or priorities. If the answer is not explicitly in the context, provide a reasoned response based on the available data.
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    rag_chain = RetrievalQA.from_llm(
        llm=groq_llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        prompt=prompt,
        return_source_documents=True
    )
    print("RAG chain with Groq and Chroma set up.")
    return rag_chain, None

# Initialize the pipeline
def initialize_pipeline(pdf_path):
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    print("HuggingFace embeddings initialized.")
    text_chunks, error = load_and_chunk_text(pdf_path)
    if error:
        return None, error
    vectorstore, error = create_vectorstore(text_chunks, embeddings)
    if error:
        return None, error
    rag_chain, error = setup_rag_chain(vectorstore)
    if error:
        return None, error
    return rag_chain, None

if __name__ == "__main__":
    pdf_path = "Budget_in_Brief.pdf"
    rag_chain, error = initialize_pipeline(pdf_path)
    if error:
        print(error)
    else:
        query = "What is the fiscal deficit for FY 2024-25?"
        result = rag_chain({"query": query})
        print(f"\nQuestion: {query}")
        print(f"Answer: {result['result']}")
        for doc in result['source_documents']:
            print(doc.page_content[:100] + "...")
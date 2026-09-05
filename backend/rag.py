import os 
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

# ── Created ONCE, reused everywhere ──────────────────────────────────
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, max_output_tokens=1024)

prompt = PromptTemplate(
    template="""You are a helpful assistant. Use the following context to answer the question.
If the answer is not in the context, say "I don't know based on the provided document."

Context: {context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

def process_pdf(file_path: str):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Fix broken text
    for doc in docs:
        doc.page_content = " ".join(doc.page_content.split())

    print(f"Loaded {len(docs)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # Windows-safe way to reset ChromaDB
    import chromadb
    client = chromadb.PersistentClient(path="my_chroma_db")

    try:
        client.delete_collection("sample")  # wipe old data
        print("Deleted old collection.")
    except:
        pass  # didn't exist, no problem

    vector_store = Chroma(
        client=client,
        embedding_function=embeddings,
        collection_name="sample"
    )
    vector_store.add_documents(chunks)
    print(f"Created ChromaDB with {vector_store._collection.count()} chunks.")

    return vector_store.as_retriever(search_kwargs={"k": 5})


def get_answer(query: str, retriever) -> str:
    retrieved_docs = retriever.invoke(query)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    final_prompt = prompt.invoke({"context": context_text, "question": query})
    response = model.invoke(final_prompt)
    return response.content


# ── Test it ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    retriever = process_pdf("dl-curriculum.pdf")
    answer = get_answer("What topics are covered in the curriculum?", retriever)
    print(f"Answer: {answer}")
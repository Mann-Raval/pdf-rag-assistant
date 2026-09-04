import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

#PDF LODADER
loader = PyPDFLoader("dl-curriculum.pdf")
docs = loader.load()
print(f"Loaded {len(docs)} documents from the PDF.")
# print(f"First document content: {docs[0].page_content[:500]}...")  # Print first 500 characters of the first document
# print(f"First document metadata: {docs[0].metadata}")

#TEXT SPLITTER
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])
chunks = text_splitter.split_documents(docs)
# print(f"Split into {len(chunks)} chunks.")
# print(f"First chunk content: {chunks[0]}")  

#Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

#Vector Store
if os.path.exists("my_chroma_db"):
    # DB already exists — just load it, don't add documents again
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="my_chroma_db",
        collection_name="sample"
    )
    print("Loaded existing ChromaDB.")
else:
    # First time — create DB and add chunks
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory="my_chroma_db",
        collection_name="sample"
    )
    vector_store.add_documents(chunks)
    print(f"Created ChromaDB with {vector_store._collection.count()} chunks.")

#Retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

#Augmentation
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, max_output_tokens=512)

user_query = "How long this course is and what are the prerequisites?"
retrieved_docs  = retriever.invoke(user_query)
context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
print(f"\nRetrieved {len(retrieved_docs)} chunks as context.")

prompt = PromptTemplate(
    template="""You are a helpful assistant. Use the following context to answer the question.
If the answer is not in the context, say "I don't know based on the provided document."

Context: {context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

final_prompt = prompt.invoke({"context": context_text, "question": user_query})

#Generation
response = model.invoke(final_prompt)
print(f"\nQuestion: {user_query}")
print(f"Answer: {response.content}")
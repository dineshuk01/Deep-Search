from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()


#PDF Read/Load
pdf_path = Path(__file__).parent/"Hands_On_ML.pdf"
loader = PyPDFLoader(file_path=pdf_path)
docs = loader.load()


#Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 400
)

split_docs = text_splitter.split_documents(documents = docs)


# Vector Embedding
embedding_model = OpenAIEmbeddings(
    model = "text-embedding-3-large"

)

# Using [embedding_model] create embeddings of [split_docs] and store in DB
vector_store = QdrantVectorStore.from_documents(
    documents = split_docs,
    url = "http://localhost:6333",
    collection_name = "learning_ML",
    embedding = embedding_model
)


print("Indexing of documents is done")






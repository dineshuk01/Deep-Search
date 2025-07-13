
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()


embedding_model = OpenAIEmbeddings(
    model = "text-embedding-3-large"
)

# Vector Similarity Seach in DB
vector_db = QdrantVectorStore.from_existing_collection(
    url = "http://localhost:6333",
    collection_name = "learning_ML",
    embedding = embedding_model
)

# Take User Query

query = input("> ")

search_results = vector_db.similarity_search(
    query=query
)

print("search results : ", search_results)
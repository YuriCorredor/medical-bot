from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")
INDEX_NAME = os.environ.get("INDEX_NAME")

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pinecone.init(
  api_key=PINECONE_API_KEY,
  environment=PINECONE_API_ENV
)

docsearch=Pinecone.from_texts(
  [t.page_content for t in text_chunks],
  embeddings,
  index_name=INDEX_NAME
)

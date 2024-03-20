import os
import pinecone
from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import prompt_template


app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")
INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")

embeddings = download_hugging_face_embeddings()

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

docsearch = Pinecone.from_existing_index(embedding=embeddings, index_name=INDEX_NAME)

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = { "prompt": PROMPT }

llm = CTransformers(
  model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
  model_type="llama",
  config={ "max_new_tokens":512, "temperature":0.8 }
)

qa = RetrievalQA.from_chain_type(
  llm=llm,
  chain_type="stuff",
  retriever=docsearch.as_retriever(search_kwargs={"k": 2}),
  return_source_documents=True,
  chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
  return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
  msg = request.form["msg"]
  input = msg
  print(input)
  result = qa({ "query": input })
  print("Response : ", result["result"])
  return str(result["result"])

if __name__ == "__main__":
  app.run(host="0.0.0.0", port=8080, debug=True)

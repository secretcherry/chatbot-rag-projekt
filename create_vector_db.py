from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

documents = []

docs_folder = "docs"

for file in os.listdir(docs_folder):

    if file.endswith(".txt"):

        loader = TextLoader(os.path.join(docs_folder, file))

        documents.extend(loader.load())

embeddings = HuggingFaceEmbeddings()

db = FAISS.from_documents(documents, embeddings)

db.save_local("vector_db")

print("Vector database created!")
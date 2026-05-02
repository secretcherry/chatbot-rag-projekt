from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

#ucitavanje dokumenta
loader = TextLoader("docs/python_docs.txt", encoding="utf-8")
documents = loader.load()

#podijela teksta u manje dijelove
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

#embedding
embeddings = HuggingFaceEmbeddings()

#vector baza
db = FAISS.from_documents(docs, embeddings)

db.save_local("vector_db")

print("Vector database created!")
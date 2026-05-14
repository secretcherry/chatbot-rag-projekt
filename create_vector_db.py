from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = TextLoader("docs/python_docs.txt", encoding="utf-8")

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=50
)

split_docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

batch_size = 100

db = None

for i in range(0, len(split_docs), batch_size):

    batch = split_docs[i:i + batch_size]

    print(f"Embedding batch {i // batch_size + 1}")

    if db is None:

        db = FAISS.from_documents(batch, embeddings)

    else:

        new_db = FAISS.from_documents(batch, embeddings)

        db.merge_from(new_db)

db.save_local("vector_db")

print("Vector database created!")
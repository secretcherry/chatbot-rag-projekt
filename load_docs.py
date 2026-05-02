from langchain_community.document_loaders import WebBaseLoader

# lista URL-ova
urls = [
    "https://docs.python.org/3/tutorial/introduction.html",
    "https://docs.python.org/3/tutorial/controlflow.html",
    "https://docs.python.org/3/tutorial/datastructures.html"
]

all_docs = []

for url in urls:
    loader = WebBaseLoader(url)
    docs = loader.load()
    all_docs.extend(docs)

# spremi u file
with open("docs/python_docs.txt", "w", encoding="utf-8") as f:
    for doc in all_docs:
        f.write(doc.page_content)
        f.write("\n\n")

print("Documentation is downloaded!")
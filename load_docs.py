import os
from bs4 import BeautifulSoup

docs_folder = "python_docs"

all_text = ""

# prolazi kroz sve html fileove
for root, dirs, files in os.walk(docs_folder):

    for file in files:

        if file.endswith(".html"):

            path = os.path.join(root, file)

            try:

                with open(path, "r", encoding="utf-8") as f:

                    html = f.read()

                    soup = BeautifulSoup(html, "html.parser")

                    # uzmi samo text
                    text = soup.get_text(separator=" ")

                    # spremi source
                    all_text += f"\nSOURCE: {path}\n"

                    all_text += text

                    all_text += "\n\n"

                    print(f"Loaded: {path}")

            except Exception as e:

                print(f"Error loading {path}: {e}")

# spremi sve u jedan file
with open("docs/python_docs.txt", "w", encoding="utf-8") as f:

    f.write(all_text)

print("All documentation loaded!")
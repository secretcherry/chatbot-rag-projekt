import os
import re
from pathlib import Path
from bs4 import BeautifulSoup

DOCS_FOLDER = "python_docs"
OUTPUT_FOLDER = "docs/chunks"

SKIP_DIRS = {
    "_static",
    "_images",
    "_sources",
    "_downloads",
    "c-api",        
    "distributing", 
    "extending",    
    "installing",   
}

SKIP_FILENAME_PATTERNS = [
    r"^genindex",
    r"^404\.html$",
    r"^about\.html$",
    r"^bugs\.html$",
    r"^copyright\.html$",
    r"^download\.html$",
    r"^contents\.html$",
]

KNOWN_SECTIONS = [
    "tutorial",
    "library",
    "reference",
    "howto",
    "faq",
    "whatsnew",
    "using",
]

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)

total = 0
skipped = 0
saved = 0

print("=" * 50)
print("Loading Python documentation")
print(f"Input:  {DOCS_FOLDER}/")
print(f"Output: {OUTPUT_FOLDER}/")
print("=" * 50)

for root, dirs, files in os.walk(DOCS_FOLDER):

    dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith("_")]

    skip_this_dir = False
    for part in Path(root).parts:
        if part in SKIP_DIRS:
            skip_this_dir = True
            break

    if skip_this_dir:
        continue

    for filename in sorted(files):
        if not filename.endswith(".html"):
            continue

        total += 1

        skip_this_file = False
        for pattern in SKIP_FILENAME_PATTERNS:
            if re.match(pattern, filename, re.IGNORECASE):
                skip_this_file = True
                break

        if skip_this_file:
            skipped += 1
            continue

        filepath = os.path.join(root, filename)
        rel_path = os.path.relpath(filepath, DOCS_FOLDER)

        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                html = f.read()

            soup = BeautifulSoup(html, "html.parser")

            for tag in soup.find_all(["nav", "footer", "script", "style"]):
                tag.decompose()

            for tag in soup.find_all(class_=["sphinxsidebar", "related", "footer",
                                              "navigation", "breadcrumbs", "headerlink"]):
                tag.decompose()

            title = ""
            h1_tag = soup.find("h1")
            if h1_tag:
                title = h1_tag.get_text(strip=True).replace("¶", "").strip()

            if not title:
                title_tag = soup.find("title")
                if title_tag:
                    title = title_tag.get_text(strip=True).split("—")[0].strip()

            content = (
                soup.find("div", class_="body")
                or soup.find("article")
                or soup.find("main")
                or soup.find("div", role="main")
                or soup.find("body")
            )

            if not content:
                skipped += 1
                continue

            text_parts = []

            for elem in content.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "code"]):
                text = elem.get_text(separator=" ", strip=True).replace("¶", "").strip()

                if text.startswith("Question"):
                    continue

                if text.startswith("Answer"):
                    continue

                if text.startswith("Next topic"):
                    continue

                if len(text) < 5:
                    continue

                if not text or len(text) < 3:
                    continue

                if elem.name == "h1":
                    text_parts.append(f"\n# {text}\n")
                elif elem.name == "h2":
                    text_parts.append(f"\n## {text}\n")
                elif elem.name == "h3":
                    text_parts.append(f"\n### {text}\n")
                elif elem.name == "h4":
                    text_parts.append(f"\n#### {text}\n")
                elif elem.name == "pre":
                    text_parts.append(f"\n[CODE]\n{text}\n[/CODE]\n")
                else:
                    text_parts.append(text)

            full_text = "\n".join(text_parts)

            full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

            if len(full_text) < 200:
                skipped += 1
                continue

            section = "general"
            for part in Path(rel_path).parts:
                if part in KNOWN_SECTIONS:
                    section = part
                    break


            stem = Path(filename).stem
            out_filename = f"{section}__{stem}.txt"
            out_filepath = Path(OUTPUT_FOLDER) / out_filename

            with open(out_filepath, "w", encoding="utf-8") as f:
                f.write(f"SECTION: {section}\n")
                f.write(f"TITLE: {title}\n")
                f.write(f"SOURCE: {rel_path}\n")
                f.write("-" * 50 + "\n\n")
                f.write(full_text)
                f.write("\n")

            saved += 1
            print(f"OK [{section}] {filename}")

        except Exception as e:
            print(f"Error: {filepath} -> {e}")
            skipped += 1

print()
print("=" * 50)
print(f"Total HTML files: {total}")
print(f"Saved:            {saved}")
print(f"Skipped:          {skipped}")
print("=" * 50)
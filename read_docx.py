import sys
from docx import Document

for file in sys.argv[1:]:
    print(f"\n--- {file} ---")
    try:
        doc = Document(file)
        for p in doc.paragraphs:
            print(p.text)
    except Exception as e:
        print(f"Error reading {file}: {e}")

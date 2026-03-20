def load_pdf(file_path):
    from pypdf import PdfReader
    import re

    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        text += page.extract_text() + "\n"

    # 🔥 CLEAN TEXT
    text = re.sub(r'\s+', ' ', text)   # remove extra spaces
    return text
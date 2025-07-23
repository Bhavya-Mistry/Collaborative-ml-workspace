import os
import pdfplumber
import docx
import tempfile

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text.strip()

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs]).strip()

def parse_resume(uploaded_file):
    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
    suffix = file_ext if file_ext in [".pdf", ".docx"] else ""

    # Use mkstemp to avoid permission errors
    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(uploaded_file.read())

        if file_ext == ".pdf":
            return extract_text_from_pdf(temp_path)
        elif file_ext == ".docx":
            return extract_text_from_docx(temp_path)
        else:
            return "Unsupported file type. Please upload a PDF or DOCX."
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

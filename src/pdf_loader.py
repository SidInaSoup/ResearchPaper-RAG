"""
PDF text extraction using PyMuPDF.
"""

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF file, returning a list of page-level text objects.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts with keys 'page' (1-indexed) and 'text'.
    """
    pages = []
    doc = fitz.open(pdf_path)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "page": page_num + 1,
                "text": text.strip(),
            })
    doc.close()
    return pages

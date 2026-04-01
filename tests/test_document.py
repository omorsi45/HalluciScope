from pathlib import Path
from backend.core.document import parse_document


def test_parse_txt_from_string():
    text = "Hello world. This is a test document."
    result = parse_document(text=text)
    assert result == text


def test_parse_txt_file(tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("Content of the text file.", encoding="utf-8")
    result = parse_document(file_path=file)
    assert result == "Content of the text file."


def test_parse_pdf_file(tmp_path):
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    page = writer.pages[0]
    pdf_path = tmp_path / "test.pdf"
    writer.write(str(pdf_path))
    result = parse_document(file_path=pdf_path)
    assert isinstance(result, str)


def test_parse_requires_input():
    try:
        parse_document()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Either text or file_path" in str(e)

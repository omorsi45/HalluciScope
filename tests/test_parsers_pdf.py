import asyncio
import io
import re
import signal
import sys
import time

import pytest
from pypdf import PdfWriter

from backend.core.document import parse_document
from backend.ingest.errors import IngestError
from backend.ingest.parsers import mime_for, parse, parse_with_timeout
from backend.ingest.parsers.pdf import parse_pdf
from backend.ingest.types import PageSpan


def make_pdf(pages: list[str]) -> bytes:
    """Build a minimal PDF with one Helvetica text line per page and exact xref offsets.

    An empty string makes an empty content stream, which is what an image-only page looks
    like to a text extractor.
    """
    bodies: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    page_ids = [4 + 2 * index for index in range(len(pages))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    bodies[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("ascii")
    for page_id, text in zip(page_ids, pages, strict=True):
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii") if text else b""
        bodies[page_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_id + 1} 0 R >>"
        ).encode("ascii")
        bodies[page_id + 1] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream"
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: dict[int, int] = {}
    for number in sorted(bodies):
        offsets[number] = len(out)  # byte offset of "<n> 0 obj", recorded as we write
        out += f"{number} 0 obj\n".encode("ascii") + bodies[number] + b"\nendobj\n"
    xref_offset = len(out)
    size = max(bodies) + 1
    out += f"xref\n0 {size}\n".encode("ascii") + b"0000000000 65535 f \n"
    for number in range(1, size):
        out += f"{offsets[number]:010d} 00000 n \n".encode("ascii")
    out += f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
        "ascii"
    )
    return bytes(out)


def with_info(data: bytes, info: dict[str, str]) -> bytes:
    writer = PdfWriter(clone_from=io.BytesIO(data))
    writer.add_metadata(info)
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def encrypted(data: bytes, user_password: str) -> bytes:
    writer = PdfWriter(clone_from=io.BytesIO(data))
    # RC4 needs no optional crypto package; AES would need `cryptography`.
    writer.encrypt(user_password, algorithm="RC4-128")
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


def error_code(call, *args) -> str:
    with pytest.raises(IngestError) as excinfo:
        call(*args)
    return excinfo.value.code


# ---------------------------------------------------------------- test PDF builder


def test_make_pdf_xref_offsets_point_at_objects():
    data = make_pdf(["One.", "", "Three."])
    xref_offset = int(re.search(rb"startxref\n(\d+)\n", data).group(1))
    # Object 0 (free) + catalog, page tree, font + 3 x (page, content stream) = 10 entries.
    assert data[xref_offset:].startswith(b"xref\n0 10\n")
    entries = data[xref_offset:].split(b"\n")[2:12]
    assert len(entries) == 10
    assert entries[0] == b"0000000000 65535 f "
    for number, entry in enumerate(entries[1:], start=1):
        offset = int(entry.split()[0])
        assert data[offset:].startswith(f"{number} 0 obj\n".encode("ascii"))


# ---------------------------------------------------------------- PDF parser


def test_pdf_pages_and_offsets():
    doc = parse_pdf(make_pdf(["Refunds take 30 days.", "Exchanges take 14 days."]))
    assert (doc.parser, doc.parser_version) == ("pdf", "pdf/1")
    assert doc.text == "Refunds take 30 days.\n\nExchanges take 14 days."
    assert doc.pages == (PageSpan(1, 0, 21), PageSpan(2, 23, 46))
    assert "Exchanges" in doc.text[doc.pages[1].start : doc.pages[1].end]
    assert doc.sections == ()
    assert doc.metadata == {}


def test_pdf_page_without_text_keeps_later_page_numbers():
    doc = parse_pdf(make_pdf(["First page.", "", "Third page."]))
    assert [span.page for span in doc.pages] == [1, 3]
    assert doc.text[doc.pages[1].start : doc.pages[1].end] == "Third page."


def test_pdf_no_text():
    assert error_code(parse_pdf, make_pdf([""])) == "no_text"


def test_pdf_garbage_bad_pdf():
    assert error_code(parse_pdf, b"%PDF-1.4 garbage") == "bad_pdf"


def test_pdf_metadata():
    data = with_info(
        make_pdf(["Policy text."]),
        {"/Title": "Refund Policy", "/Author": "Ops Team", "/CreationDate": "D:20240102030405Z"},
    )
    doc = parse_pdf(data)
    assert doc.metadata["title"] == "Refund Policy"
    assert doc.metadata["author"] == "Ops Team"
    assert doc.metadata["created"].startswith("2024-01-02T03:04:05")


def test_pdf_empty_metadata_strings_are_omitted():
    doc = parse_pdf(with_info(make_pdf(["Policy text."]), {"/Title": "  ", "/Author": ""}))
    assert doc.metadata == {}


def test_pdf_metadata_is_normalized_like_body_text():
    # A NUL in a stored string would be rejected by Postgres, so control characters go.
    doc = parse_pdf(with_info(make_pdf(["Policy text."]), {"/Title": " Refund\x00  Policy "}))
    assert doc.metadata["title"] == "Refund Policy"


def test_pdf_unparsable_creation_date_is_ignored():
    # A junk date in the info dictionary is missing metadata, not a corrupt file.
    data = with_info(make_pdf(["Policy text."]), {"/CreationDate": "last tuesday"})
    doc = parse_pdf(data)
    assert doc.text == "Policy text."
    assert "created" not in doc.metadata


def test_pdf_password_protected_is_rejected():
    assert error_code(parse_pdf, encrypted(make_pdf(["Secret."]), "hunter2")) == "encrypted_pdf"


def test_pdf_empty_user_password_is_decrypted():
    doc = parse_pdf(encrypted(make_pdf(["Open to all."]), ""))
    assert doc.text == "Open to all."


# ---------------------------------------------------------------- dispatch


@pytest.mark.parametrize(
    ("filename", "mime"),
    [
        ("a.pdf", "application/pdf"),
        ("a.txt", "text/plain"),
        ("a.md", "text/markdown"),
        ("a.markdown", "text/markdown"),
        ("a.html", "text/html"),
        ("a.htm", "text/html"),
        ("a.csv", "text/csv"),
        ("A.PDF", "application/pdf"),
        ("report.v2.Csv", "text/csv"),
    ],
)
def test_mime_for_known_extensions(filename, mime):
    assert mime_for(filename) == mime


@pytest.mark.parametrize("filename", ["a.docx", "noextension", "a.pdf.exe", ""])
def test_mime_for_unknown_extension(filename):
    assert error_code(mime_for, filename) == "unsupported_type"


@pytest.mark.parametrize(
    ("filename", "data", "parser"),
    [
        ("notes.md", b"# Refunds\n\nTake 30 days.", "markdown"),
        ("notes.markdown", b"# Refunds\n\nTake 30 days.", "markdown"),
        ("PAGE.HTML", b"<p>Take 30 days.</p>", "html"),
        ("page.htm", b"<p>Take 30 days.</p>", "html"),
        ("table.csv", b"item,days\nrefund,30\n", "csv"),
        ("plain.txt", b"Take 30 days.", "text"),
        ("DOC.PDF", make_pdf(["Take 30 days."]), "pdf"),
    ],
)
def test_dispatch_by_extension(filename, data, parser):
    doc = parse(data, filename)
    assert doc.parser == parser
    assert "30" in doc.text


@pytest.mark.parametrize("filename", ["report.docx", "noextension"])
def test_dispatch_unknown_extension(filename):
    assert error_code(parse, b"Some text.", filename) == "unsupported_type"


@pytest.mark.parametrize(
    ("filename", "data"),
    [
        ("x.pdf", b"This is plain text, not a PDF."),
        ("x.pdf", b""),
        ("x.txt", b"%PDF-1.4 hidden pdf"),
        ("x.html", b"%PDF-1.4 hidden pdf"),
        ("x.csv", b"%PDF-1.4,hidden\n1,2"),
    ],
)
def test_pdf_magic_required(filename, data):
    assert error_code(parse, data, filename) == "unsupported_type"


def test_too_long():
    assert error_code(parse, ("a. " * 700_001).encode("utf-8"), "big.txt") == "too_long"


def test_exactly_at_the_character_limit_is_accepted():
    doc = parse(("a" * 2_000_000).encode("utf-8"), "edge.txt")
    assert len(doc.text) == 2_000_000


@pytest.mark.parametrize("data", [b"", b"  \n\t \n"])
def test_empty_or_blank_text_is_no_text(data):
    assert error_code(parse, data, "empty.txt") == "no_text"


# ---------------------------------------------------------------- legacy module


def test_legacy_parse_document_still_works(tmp_path):
    path = tmp_path / "legacy.txt"
    path.write_text("Legacy text still reads.", encoding="utf-8")
    assert parse_document(file_path=path) == "Legacy text still reads."


def test_legacy_parse_document_reads_pdf_text(tmp_path):
    path = tmp_path / "legacy.pdf"
    path.write_bytes(make_pdf(["Legacy pdf text."]))
    assert parse_document(file_path=path) == "Legacy pdf text."


def test_legacy_parse_document_blank_pdf_is_empty_text(tmp_path):
    path = tmp_path / "blank.pdf"
    path.write_bytes(make_pdf([""]))
    assert parse_document(file_path=path) == ""


def test_legacy_parse_document_returns_given_text_unchanged():
    assert parse_document(text="  raw  text ") == "  raw  text "


# ---------------------------------------------------------------- parse subprocess


class ChildSpy:
    """Records every spawned child; can swap the child command for `python -c <code>`."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.procs: list[asyncio.subprocess.Process] = []
        self.spawned = asyncio.Event()
        self._real_exec = asyncio.create_subprocess_exec
        self._code: str | None = None
        monkeypatch.setattr(asyncio, "create_subprocess_exec", self._spawn)

    def run_code_instead(self, code: str) -> None:
        self._code = code

    async def _spawn(self, *args, **kwargs):
        if self._code is not None:
            args = (sys.executable, "-c", self._code)
        proc = await self._real_exec(*args, **kwargs)
        self.procs.append(proc)
        self.spawned.set()
        return proc


@pytest.fixture
def child_spy(monkeypatch):
    return ChildSpy(monkeypatch)


@pytest.mark.parametrize(
    ("filename", "data"),
    [
        ("plain.txt", "Café ☕ 😀 漢字, e\u0301 combining.\n\nSecond paragraph.".encode()),
        ("policy.md", b"# Refunds\n\nTake 30 days.\n\n## Exceptions\n\nNone apply."),
        ("table.csv", b"item,days\nrefund,30\nexchange,14\n"),
        (
            "policy.pdf",
            with_info(
                make_pdf(["Refunds take 30 days.", "Exchanges take 14 days."]),
                {"/Title": "Refund Policy"},
            ),
        ),
    ],
)
async def test_parse_with_timeout_roundtrip(filename, data):
    expected = parse(data, filename)
    assert await parse_with_timeout(data, filename) == expected


def test_roundtrip_inputs_include_pages_and_sections():
    # Guards the roundtrip test above against comparing two empty span tuples.
    assert parse(make_pdf(["One.", "Two."]), "a.pdf").pages
    assert parse(b"# Heading\n\nBody.", "a.md").sections


@pytest.mark.parametrize(
    ("filename", "data", "code"),
    [
        ("empty.txt", b"", "no_text"),
        ("latin1.txt", b"caf\xe9", "bad_encoding"),
        ("garbage.pdf", b"%PDF-1.4 garbage", "bad_pdf"),
        ("fake.pdf", b"not a pdf at all", "unsupported_type"),
        ("report.docx", b"PK\x03\x04", "unsupported_type"),
    ],
)
async def test_parse_with_timeout_error_code(filename, data, code):
    with pytest.raises(IngestError) as excinfo:
        await parse_with_timeout(data, filename)
    assert excinfo.value.code == code


async def test_parse_with_timeout_kills(child_spy):
    started = time.monotonic()
    with pytest.raises(IngestError) as excinfo:
        await parse_with_timeout(b"Some text.", "a.txt", timeout_s=0.001)
    assert excinfo.value.code == "parse_timeout"
    assert time.monotonic() - started < 5
    (proc,) = child_spy.procs
    assert proc.returncode == -signal.SIGKILL  # killed, and reaped (returncode is only set then)


async def test_parse_with_timeout_kills_a_hung_child_that_never_reads_stdin(child_spy):
    child_spy.run_code_instead("import time; time.sleep(600)")
    started = time.monotonic()
    with pytest.raises(IngestError) as excinfo:
        # Bigger than the pipe buffer, so the parent is blocked writing when the timer fires.
        await parse_with_timeout(b"a" * (4 << 20), "a.txt", timeout_s=0.5)
    assert excinfo.value.code == "parse_timeout"
    assert time.monotonic() - started < 10
    (proc,) = child_spy.procs
    assert proc.returncode == -signal.SIGKILL


async def test_parse_with_timeout_kills_the_child_when_cancelled(child_spy):
    child_spy.run_code_instead("import time; time.sleep(600)")
    task = asyncio.create_task(parse_with_timeout(b"Some text.", "a.txt", timeout_s=600))
    await asyncio.wait_for(child_spy.spawned.wait(), timeout=10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert child_spy.procs[0].returncode == -signal.SIGKILL


@pytest.mark.parametrize(
    "code",
    [
        "import sys; sys.exit(3)",
        "print('not json')",
        "print('[1, 2]')",
        "print('{\"ok\": true}')",
        "print('{\"ok\": true, \"doc\": {\"text\": \"x\"}}')",
        "print('{\"ok\": false, \"code\": \"not_a_real_code\"}')",
    ],
)
@pytest.mark.parametrize(
    ("filename", "code_for_failure"),
    [("broken.pdf", "bad_pdf"), ("broken.txt", "unsupported_type")],
)
async def test_parse_with_timeout_broken_child_maps_to_error(
    child_spy, code, filename, code_for_failure
):
    child_spy.run_code_instead(code)
    with pytest.raises(IngestError) as excinfo:
        await parse_with_timeout(b"%PDF-1.4 whatever", filename)
    assert excinfo.value.code == code_for_failure

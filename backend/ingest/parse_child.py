"""Child process entry point: parse one untrusted file in isolation.

Usage: `python -m backend.ingest.parse_child <filename>` with the file bytes on stdin.
Writes one JSON object to stdout and exits 0 for both outcomes:

    {"ok": true, "doc": {text, pages: [[page, start, end]], sections: [[start, end, path]],
                         metadata, parser, parser_version}}
    {"ok": false, "code": "<ingest error code>"}

Any other exit (crash, memory exhaustion, kill) is reported by the parent. Stderr is
discarded by the parent, so a traceback never carries document content into a log. This
module imports only `backend.ingest`, the standard library, and pypdf (through the parsers).
"""

import json
import sys

from backend.ingest.errors import IngestError
from backend.ingest.parsers import parse


def main(argv: list[str]) -> int:
    filename = argv[1]
    data = sys.stdin.buffer.read()
    try:
        doc = parse(data, filename)
    except IngestError as exc:
        result: dict = {"ok": False, "code": exc.code}
    else:
        result = {
            "ok": True,
            "doc": {
                "text": doc.text,
                "pages": [[span.page, span.start, span.end] for span in doc.pages],
                "sections": [[span.start, span.end, span.path] for span in doc.sections],
                "metadata": doc.metadata,
                "parser": doc.parser,
                "parser_version": doc.parser_version,
            },
        }
    # ASCII-only JSON (the default) cannot fail on stdout's encoding, even for lone surrogates.
    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

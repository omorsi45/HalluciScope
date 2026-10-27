from backend.ingest.segment import segment_sentences


def texts(s: str) -> list[str]:
    return [s[x.start : x.end] for x in segment_sentences(s)]


def test_basic_split():
    assert texts("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_abbreviations_and_initials():
    result = texts("Dr. Smith met J. R. Tolkien at 3 p.m. on Main St. today. Next.")
    assert len(result) == 2
    assert result[0].endswith("today.")


def test_decimal_not_split():
    assert len(texts("Pi is 3.14 today. Yes.")) == 2


def test_quotes_close_sentence():
    assert texts('He said "Stop." Then left.') == ['He said "Stop."', "Then left."]


def test_paragraph_and_wrap():
    assert texts("Line one wraps\nhere. Next para\n\nSecond para") == [
        "Line one wraps\nhere.",
        "Next para",
        "Second para",
    ]


def test_bullets_are_sentences():
    assert len(texts("Rules:\n- first item\n- second item")) == 3


def test_offsets_with_range_and_unicode():
    s = "Skip. Café 😀 opens. Done."
    sentences = segment_sentences(s, 6)
    assert [s[x.start : x.end] for x in sentences] == ["Café 😀 opens.", "Done."]
    assert [(x.start, x.end) for x in sentences] == [
        (s.index("Café"), s.index(" Done")),
        (s.index("Done"), len(s)),
    ]


def test_empty_and_whitespace():
    assert segment_sentences("") == []
    assert segment_sentences("  \n ") == []


def test_lowercase_after_period_not_split():
    assert len(texts("See fig. 2 and the e.g. case. End.")) == 2


def test_numbered_list_marker_is_not_a_sentence():
    assert texts("Steps:\n1. First step\n2. Second step") == [
        "Steps:",
        "1. First step",
        "2. Second step",
    ]


def test_end_bound_limits_scan():
    s = "One. Two. Three."
    sentences = segment_sentences(s, 0, 9)
    assert [(x.start, x.end) for x in sentences] == [(0, 4), (5, 9)]

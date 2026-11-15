import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from storage.models import ResearchPaperData  # noqa: E402
from utils.citation_meta import merge_into_paper, parse_citation_meta  # noqa: E402

ARXIV_HEAD = """<html><head>
<meta name="citation_title" content="Attention Is All You Need" />
<meta name="citation_author" content="Vaswani, Ashish" />
<meta name="citation_author" content="Shazeer, Noam" />
<meta name="citation_author" content="Jones, Llion" />
<meta name="citation_date" content="2017/06/12" />
<meta name="citation_pdf_url" content="https://arxiv.org/pdf/1706.03762" />
<meta name="citation_arxiv_id" content="1706.03762" />
<meta name="citation_doi" content="https://doi.org/10.48550/arXiv.1706.03762" />
</head><body>From: Llion Jones</body></html>"""


def test_parse_citation_meta():
    m = parse_citation_meta(ARXIV_HEAD)
    assert m["title"] == "Attention Is All You Need"
    assert m["authors"] == ["Vaswani, Ashish", "Shazeer, Noam", "Jones, Llion"]
    assert m["year"] == 2017 and m["arxiv_id"] == "1706.03762"
    assert m["doi"] == "10.48550/arXiv.1706.03762"


def test_no_meta():
    assert parse_citation_meta("<html><head><title>x</title></head></html>") == {}
    assert parse_citation_meta(None) == {}


def test_merge_meta_authors_win_and_gaps_filled():
    llm = ResearchPaperData(title="Attention Is All You Need", authors=["Llion Jones"],
                            year=2017, confidence=0.9)
    merged = merge_into_paper(llm, parse_citation_meta(ARXIV_HEAD))
    assert len(merged.authors) == 3 and merged.authors[0] == "Vaswani, Ashish"
    assert merged.title == "Attention Is All You Need"
    assert merged.pdf_url == "https://arxiv.org/pdf/1706.03762"
    assert merged.doi == "10.48550/arXiv.1706.03762"
    assert merged.confidence == 0.9

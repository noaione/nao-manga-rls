"""
MIT License

Copyright (c) 2022-present noaione

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from nmanga.common import RegexCollection


def test_volume_regex():
    remedy = RegexCollection.volume_re("Real Manga")

    assert remedy.match("Real Manga 1") is None
    assert remedy.match("Real Manga v01") is not None
    assert remedy.match("Real Manga v01 (2023) (Digital) (nao)") is not None

    assert remedy.pattern == r"Real\ Manga v(\d+).*"


def test_volume_regex_with_limiter():
    remedy = RegexCollection.volume_re("Real Manga", "danke")

    assert remedy.match("Real Manga 1") is None
    assert remedy.match("Real Manga v01") is None
    assert remedy.match("Real Manga v01 (2023) (Digital) (nao)") is None
    assert remedy.match("Real Manga v01 (2023) (Digital) (danke)") is not None
    assert remedy.match("Real Manga v01 (2023) (Digital) (danke-Empire)") is not None

    assert remedy.pattern == r"Real\ Manga v(\d+).*[\[\(]danke.*"


def test_chapter_regex():
    remedy = RegexCollection.chapter_re("Real Manga")

    assert remedy.match("Real Manga - c001 (v01) - p001") is not None
    assert remedy.match("Real Manga - c001 (v01) - p001-002") is not None
    assert remedy.match("Real Manga - c001 (OShot) - p001") is not None
    assert remedy.match("Real Manga - c001 (OShot) - p001-002") is not None
    assert remedy.match("Real Manga - c001x1 (v01) - p001") is not None
    assert remedy.match("Real Manga - c001x1 (v01) - p001-002") is not None
    assert remedy.match("Real Manga - c001x1 (OShot) - p001") is not None
    assert remedy.match("Real Manga - c001x1 (OShot) - p001-002") is not None
    assert remedy.match("Real Manga - c001x1 (c000.1) (v01) - p001") is not None
    assert remedy.match("Real Manga - c001x1 (c000.1) (v01) - p001-002") is not None
    assert remedy.match("Real Manga - c001x1 (c000.1) (OShot) - p001") is not None
    assert remedy.match("Real Manga - c001x1 (c000.1) (OShot) - p001-002") is not None

    assert (
        remedy.pattern == r"Real\ Manga - c(?P<ch>\d+)(?P<ex>[\#x.][\d]{1,2})? \(?c?(?P<actual>[\d]{1,3}[\.][\d]{1,3})"
        r"?\)? ?\(?(?P<vol>v[\d]+(?P<volex>[\#x][\d]{1,2})?|[Oo][Ss]hot|[Oo]ne[ -]?[Ss]hot|[Nn][Aa])?\)? ?- "
        r"p[\d]+x?[\d]?\-?[\d]+x?[\d]?.*"
    )


def test_chapter_regex_with_publisher():
    remedy = RegexCollection.chapter_re("Real Manga", "Kodansha")

    backremedy = "[dig] [Kodansha Comics] [nao]"
    backremedytitle = "[dig] [Introduction] [Kodansha Comics] [nao]"
    invalidmatch = "[dig] [Seven Seas] [nao]"
    invalidmatchtitle = "[dig] [Introduction] [Seven Seas] [nao]"

    assert remedy.match(f"Real Manga - c001 (v01) - p001 {backremedy}") is not None
    assert remedy.match(f"Real Manga - c001 (v01) - p001-002 {backremedytitle}") is not None
    assert remedy.match(f"Real Manga - c001 (OShot) - p001 {backremedy}") is not None
    assert remedy.match(f"Real Manga - c001x1 (v01) - p001 {backremedy}") is not None
    assert remedy.match(f"Real Manga - c001x1 (c000.1) (v01) - p001 {backremedy}") is not None
    assert remedy.match(f"Real Manga - c001x1 (c000.1) (v01) - p001-002 {backremedytitle}") is not None
    assert remedy.match(f"Real Manga - c001 (v01) - p001-002 {invalidmatch}") is None
    assert remedy.match(f"Real Manga - c001 (v01) - p001-002 {invalidmatchtitle}") is None

    assert (
        remedy.pattern == r"Real\ Manga - c(?P<ch>\d+)(?P<ex>[\#x.][\d]{1,2})? \(?c?(?P<actual>[\d]{1,3}[\.][\d]"
        r"{1,3})?\)? ?\(?(?P<vol>v[\d]+(?P<volex>[\#x][\d]{1,2})?|[Oo][Ss]hot|[Oo]ne[ -]?[Ss]hot|[Nn][Aa])?\)? ?- "
        r"p[\d]+x?[\d]?\-?[\d]+x?[\d]?.* "
        r"\[(?:dig|web|c2c|mag|scan|paper|raw|raw-d|raw-dig|raw-digital|raw-m|raw-mag|raw-magazine)] "
        r"(?:\[(?P<title>.*)\] )?\[Kodansha.*"
    )


def test_comix_styled_regex():
    cmx_re = RegexCollection.cmx_re()

    assert cmx_re.match("Real Manga - v01 - p000") is not None
    assert cmx_re.match("Real Manga - p000") is not None
    assert cmx_re.match("Real Manga - v01 - x000") is None
    assert cmx_re.match("What even is this format") is None


def test_page_styled_regex():
    page_re = RegexCollection.page_re()

    assert page_re.match("Real Manga - v01 - p000") is not None
    assert page_re.match("Real Manga - p000") is not None
    assert page_re.match("Real Manga - v01 - x000") is None
    assert page_re.match("test - p001") is not None
    assert page_re.match("test - p001 - thing") is not None
    assert page_re.match("p001-002 - thing") is not None
    assert page_re.match("What even is this format") is None

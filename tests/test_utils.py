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

from nmanga.utils import secure_filename, clean_title, is_oneshot, decode_or, encode_or


class TestSecureFilename:
    def test_no_changes(self):
        safe_fn = secure_filename("test")
        assert safe_fn == "test"

    def test_replacement(self):
        safe_fn = secure_filename("test: test")
        assert safe_fn == "test： test"

    def test_emoji(self):
        safe_fn = secure_filename("test: test 🤔")
        assert safe_fn == "test： test _"


class TestCleanTitle:
    def test_no_changes(self):
        title = clean_title("test")
        assert title == "test"

    def test_remove_bracket(self):
        title = clean_title("test [test]")
        assert title == "test [test"

    def test_empty(self):
        title = clean_title("")
        assert title == ""


class TestIsOneshot:
    def test_oneshot(self):
        assert is_oneshot("oneshot")
        assert is_oneshot("oshot")
        assert is_oneshot("NA")
        assert is_oneshot("ONE SHOT")

    def test_not_oneshot(self):
        assert not is_oneshot("test")


class TestDecodeOr:
    def test_decode(self):
        assert decode_or("test") == "test"

    def test_decode_none(self):
        assert decode_or(None) is None

    def test_decode_bytes(self):
        assert isinstance(decode_or(b"test"), str)


class TestEncodeOr:
    def test_encode(self):
        assert encode_or("test") == b"test"

    def test_encode_none(self):
        assert encode_or(None) is None

    def test_encode_bytes(self):
        assert not isinstance(encode_or(b"test"), str)

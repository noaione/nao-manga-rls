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

# Contains pure constants string data for .epub document

EPUB_CONTAINER = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>
"""  # noqa

EPUB_CONTENT = """<?xml version='1.0' encoding='utf-8'?>
<package version="3.0" xml:lang="en" xmlns="http://www.idpf.org/2007/opf" prefix="rendition: http://www.idpf.org/vocab/rendition/# fixed-layout-jp: http://www.digital-comic.jp ibooks: http://vocabulary.itunes.apple.com/rdf/ibooks/vocabulary-extensions-1.0/" unique-identifier="pub-id" dir="rtl">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:epub="http://www.idpf.org/2011/epub">
        <dc:title id="id">{title}</dc:title>
        <dc:language>en</dc:language>
        <dc:identifier id="pub-id">{identifier}</dc:identifier>
        <meta property="dcterms:modified">{time}</meta>
        <dc:date>{time}</dc:date>

        <meta name="cover" content="image-cover"/>
        <meta name="zero-gutter" content="true"/>
        <meta name="zero-margin" content="true"/>
        <meta name="RegionMagnification" content="false"/>
        <meta name="SpineColor" content="#FFFFFF"/>
        <meta name="fixed-layout" content="true"/>
        <meta name="orientation-lock" content="none"/>
        <meta name="book-type" content="comic"/>
        <meta name="primary-writing-mode" content="horizontal-rl"/>
        <meta property="ibooks:binding">false</meta>
        <meta property="rendition:layout">pre-paginated</meta>
        <meta property="rendition:orientation">auto</meta>
        <meta property="rendition:spread">landscape</meta>
    </metadata>
    <manifest>

    </manifest>
    <spine toc="ncx" page-progression-direction="rtl">

    </spine>
</package>
"""  # noqa

EPUB_PAGE = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml">
    <head>
        <title>{title}</title>
        <meta name="viewport" content="width={width}, height={height}"/>
        <link rel="stylesheet" type="text/css" href="../Styles/styles.css"/>
    </head>
    <body>
        <svg xmlns="http://www.w3.org/2000/svg" version="1.1" xmlns:xlink="http://www.w3.org/1999/xlink" width="100%" height="100%" viewBox="0 0 {width} {height}">
            <image width="{width}" height="{height}" xlink:href="../Images/{filename}"/>
        </svg>
    </body>
</html>
"""  # noqa

EPUB_STYLES = """@charset "UTF-8";
html, body { width: 100%; height: 100%; margin: 0; padding: 0; font-size: 0; }
svg { margin: 0; padding: 0; }
img { margin: 0; padding: 0; border: 0; }
p { display: none; }
"""  # noqa

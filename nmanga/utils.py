import re

__all__ = ("secure_filename", "clean_title", "is_oneshot")


def secure_filename(fn: str):
    replacement = {
        "/": "／",
        ":": "：",
        "<": "＜",
        ">": "＞",
        '"': "”",
        "'": "’",
        "\\": "＼",
        "?": "？",
        "*": "⋆",
        "|": "｜",
        "#": "",
    }
    for k, v in replacement.items():
        fn = fn.replace(k, v)
    EMOJI_PATTERN = re.compile(
        "(["
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "])"
    )
    fn = re.sub(EMOJI_PATTERN, "_", fn)
    return fn


def clean_title(title: str):
    if not title:
        return title
    if title.endswith("]"):
        title = title[:-1]
    return title


def is_oneshot(title: str):
    title = title.lower()
    valid_oshot = [
        "na",
        "oshot",
        "oneshot",
        "one-shot",
        "one shot",
    ]
    return title in valid_oshot

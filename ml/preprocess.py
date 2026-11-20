"""Text cleaning - mirrors the cleaning steps from the original notebook.

Notebook steps: remove @mentions -> keep only letters/# -> drop short words
(len <= 3) -> (lemmatize). Lemmatization needs a WordNet download, so it is
left out here to keep the project fully offline; TF-IDF handles most variants.
"""
import re

MIN_WORD_LEN = 4  # notebook kept words with len > 3

_MENTION = re.compile(r"@\w*")
_NON_ALPHA = re.compile(r"[^a-zA-Z#]")


def normalize_token(token: str) -> str:
    """Lower-case a raw token and strip everything except letters and '#'."""
    return _NON_ALPHA.sub("", token).lower()


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = _MENTION.sub(" ", text)
    text = _NON_ALPHA.sub(" ", text)
    words = [w for w in text.split() if len(w) >= MIN_WORD_LEN]
    return " ".join(words).lower()

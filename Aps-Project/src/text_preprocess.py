"""Text normalisation for English/German support tickets."""
import re

_URL = re.compile(r"https?://\S+|www\.\S+")
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_NUM = re.compile(r"\b\d[\d.,:/-]*\b")
_PLACEHOLDER = re.compile(r"<[^>]{1,40}>|\{[^}]{1,40}\}")  # e.g. <name>, {product}
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = text.replace("\\n", " ")
    text = text.lower()
    text = _URL.sub(" urltoken ", text)
    text = _EMAIL.sub(" emailtoken ", text)
    text = _PLACEHOLDER.sub(" placeholder ", text)
    text = _NUM.sub(" numtoken ", text)
    return _WS.sub(" ", text).strip()


def ticket_text(subject, body) -> str:
    """Subject is repeated so its words weigh a bit more than body words."""
    subject = "" if subject is None or subject != subject else str(subject)
    return clean_text(f"{subject} . {subject} . {body}")
